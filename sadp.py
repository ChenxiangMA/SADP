import math
import warnings
from typing import Iterator, Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, _DatasetKind
from torch.utils.data.dataloader import _BaseDataLoaderIter
from torch.utils.data.distributed import DistributedSampler


__all__ = ['SADP']


def hack_indices(self):
    with torch.autograd.profiler.record_function(self._profile_name):
        if self._sampler_iter is None:
            # TODO(https://github.com/pytorch/pytorch/issues/76750)
            self._reset()  # type: ignore[call-arg]
        if isinstance(self._dataset, SADP):
            indices, data = self._next_data()
        else:
            data = self._next_data()
        self._num_yielded += 1
        if self._dataset_kind == _DatasetKind.Iterable and \
                self._IterableDataset_len_called is not None and \
                self._num_yielded > self._IterableDataset_len_called:
            warn_msg = (
                'Length of IterableDataset {} was reported to be {} '
                '(when accessing len(dataloader)), but {} samples have been '
                'fetched. '
            ).format(
                self._dataset,
                self._IterableDataset_len_called,
                self._num_yielded,
            )
            if self._num_workers > 0:
                warn_msg += (
                    'For multiprocessing data-loading, this could be caused '
                    'by not properly configuring the IterableDataset replica '
                    'at each worker. Please see https://pytorch.org/docs/'
                    'stable/data.html#torch.utils.data.IterableDataset for '
                    'examples.'
                )
            warnings.warn(warn_msg)
        if isinstance(self._dataset, SADP):
            self._dataset.set_active_indices(indices)
        return data


_BaseDataLoaderIter.__next__ = hack_indices


@torch.no_grad()
def concat_all_gather(tensor, dim=0):
    """Gather tensors from all ranks without gradient propagation."""
    tensors_gather = [
        torch.empty_like(tensor) for _ in range(dist.get_world_size())
    ]
    dist.all_gather(tensors_gather, tensor, async_op=False)
    return torch.cat(tensors_gather, dim=dim)


class SADP(Dataset):
    def __init__(self,
                 dataset: Dataset,
                 num_epochs: int,
                 prune_ratio: float = 0.5,
                 bern_generator: Optional[torch.Generator] = None,
                 adaptive_prune_ratio: bool = False,
                 keep_ratio_gap_to_min: float = 0.1,
                 num_redistrib: int = 6,
                 smooth_constant: float = 0.0,
                 min_value: float = 1e-5,
                 device: torch.device = torch.device('cpu')):
        self.dataset = dataset
        self.keep_ratio = min(1.0, max(1e-2, 1.0 - prune_ratio))
        self.num_epochs = num_epochs
        self.device = torch.device(device)
        self.scores = torch.zeros(len(dataset), device=self.device)
        self.weights = torch.ones(len(dataset), device=self.device)
        self.num_pruned_samples = 0
        self.cur_batch_index = None
        self.use_full_data = False

        if bern_generator is None:
            bern_generator = torch.Generator(device=self.device)
            bern_generator.manual_seed(0)
        self.bern_generator = bern_generator
        self.base_seed = int(bern_generator.initial_seed())
        self.numpy_rng = np.random.default_rng(self.base_seed)

        self.adaptive_prune_ratio = adaptive_prune_ratio
        self.num_redistrib = num_redistrib
        self.min_value = min_value
        self.smooth_constant = smooth_constant
        self.smooth_ratio = smooth_constant / self.keep_ratio

        if self.adaptive_prune_ratio:
            mean_keep_ratio = self.keep_ratio
            min_keep_ratio = self.keep_ratio - keep_ratio_gap_to_min
            self.ratio_pace = (
                2 * (mean_keep_ratio - min_keep_ratio)
                / (self.num_epochs - 2)
            )
            self.max_keep_ratio = (
                mean_keep_ratio + mean_keep_ratio - min_keep_ratio
            )
            self.curr_epoch = 1

    def __getattr__(self, name):
        return getattr(self.dataset, name)

    def set_active_indices(self, cur_batch_indices: torch.Tensor):
        self.cur_batch_index = cur_batch_indices.to(
            device=self.device, dtype=torch.long, non_blocking=True
        )

    @torch.no_grad()
    def update_score(self, importance_score):
        assert isinstance(importance_score, torch.Tensor)
        assert len(self.cur_batch_index) == importance_score.shape[0], 'not enough index'
        indices = self.cur_batch_index
        importance_score = importance_score.detach().to(
            device=self.device, dtype=self.scores.dtype, non_blocking=True
        )
        if dist.is_available() and dist.is_initialized():
            indices = concat_all_gather(indices, dim=0)
            importance_score = concat_all_gather(importance_score, dim=0)
        self.scores.index_copy_(0, indices, importance_score)
        self.cur_batch_index = None

    def update(self, loss):
        assert isinstance(loss, torch.Tensor)
        assert len(self.cur_batch_index) == loss.shape[0], 'not enough index'
        weights = self.weights.index_select(0, self.cur_batch_index)
        if weights.device != loss.device:
            weights = weights.to(loss.device, non_blocking=True)
        loss.mul_(weights.detach())
        return loss.mean()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return index, self.dataset[index]

    @staticmethod
    def _assign_probs(sub_scores,
                      target_sum,
                      smooth_constant,
                      effective_keep_ratio):
        if smooth_constant > 0.0 and \
                smooth_constant < effective_keep_ratio:
            denominator = 1.0 - smooth_constant / effective_keep_ratio
            if denominator > 1e-10:
                revised_smooth_constant = (
                    smooth_constant / target_sum * sub_scores.sum()
                    - sub_scores.min()
                ) / denominator
                revised_smooth_constant.clamp_min_(0.0)
                probabilities = sub_scores + revised_smooth_constant
            else:
                probabilities = sub_scores
            return probabilities * (target_sum / probabilities.sum())
        if smooth_constant > 0.0:
            return torch.full_like(
                sub_scores, fill_value=effective_keep_ratio
            )
        return sub_scores * (target_sum / sub_scores.sum())

    @classmethod
    def _compute_initial_probs(cls,
                               scores,
                               non_zero_mask,
                               num_subset,
                               smooth_constant,
                               keep_ratio):
        probabilities = torch.zeros_like(scores)
        probabilities[non_zero_mask] = cls._assign_probs(
            scores[non_zero_mask],
            num_subset,
            smooth_constant,
            keep_ratio,
        )
        return probabilities

    def _redistribute_probs(self, probabilities, scores, num_subset):
        """Cap and redistribute probabilities for at most num_redistrib passes."""

        for redistribution in range(self.num_redistrib):
            saturated_mask = probabilities >= 1.0
            in_range_mask = (probabilities > 0.0) & (probabilities < 1.0)
            stats = torch.stack((
                probabilities.max(),
                saturated_mask.sum().to(probabilities.dtype),
                in_range_mask.sum().to(probabilities.dtype),
            )).tolist()
            max_probability = stats[0]
            if max_probability <= 1.0:
                break

            num_saturated = int(stats[1])
            num_in_range = int(stats[2])
            probabilities[saturated_mask] = 1.0
            remaining_sum = float(num_subset) - num_saturated
            if remaining_sum <= 0.0 or num_in_range == 0:
                break

            effective_keep_ratio = remaining_sum / num_in_range
            probabilities[in_range_mask] = self._assign_probs(
                scores[in_range_mask],
                remaining_sum,
                self.smooth_constant,
                effective_keep_ratio,
            )
        return probabilities

    @torch.no_grad()
    def spike_aware_prune(self, iteration):
        if iteration <= 1:
            selected_indices = list(range(len(self)))
        else:
            self.scores.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
            self.scores.clamp_min_(0.0)

            if self.adaptive_prune_ratio:
                self.keep_ratio = (
                    self.max_keep_ratio
                    - (self.curr_epoch - 1) * self.ratio_pace
                )
                self.smooth_constant = self.keep_ratio * self.smooth_ratio

            num_samples = len(self)
            num_subset = num_samples * self.keep_ratio
            scores = self.scores.detach().clone()
            non_zero_mask = scores > self.min_value
            num_non_zero = int(non_zero_mask.sum().item())

            if num_non_zero == 0:
                # All scores zero/negative: uniform selection
                per_sample_prob = torch.full_like(scores, self.keep_ratio)
            elif num_non_zero <= num_subset:
                # Fewer non-zero samples than needed: keep all non-zero, fill from zeros
                per_sample_prob = torch.zeros_like(scores)
                per_sample_prob[non_zero_mask] = 1.0
                fill_needed = num_subset - num_non_zero
                num_zero = num_samples - num_non_zero
                fill_count = min(
                    num_zero,
                    max(0, int(round(fill_needed))),
                )

                if fill_count >= num_zero:
                    per_sample_prob[~non_zero_mask] = 1.0
                elif fill_count > 0:
                    zero_indices = (~non_zero_mask).nonzero(as_tuple=True)[0]
                    sampled_positions = torch.randperm(
                        num_zero,
                        device=scores.device,
                        generator=self.bern_generator,
                    )[:fill_count]
                    per_sample_prob[zero_indices[sampled_positions]] = 1.0
            else:
                # Standard case: score-proportional probabilities
                per_sample_prob = self._compute_initial_probs(
                    scores,
                    non_zero_mask,
                    num_subset,
                    self.smooth_constant,
                    self.keep_ratio,
                )
                # Iterative redistribution of excess probability mass
                per_sample_prob = self._redistribute_probs(
                    per_sample_prob, scores, num_subset
                )

            # After at most num_redistrib passes, any remaining values above one are clipped.
            per_sample_prob.clamp_(min=0.0, max=1.0)
            selected_mask = torch.bernoulli(
                per_sample_prob, generator=self.bern_generator
            )
            selected_indices = selected_mask.nonzero(as_tuple=True)[0].tolist()

            self.weights.fill_(1.0)
            positive_prob = per_sample_prob > 0.0
            self.weights[positive_prob] = (
                self.keep_ratio / per_sample_prob[positive_prob]
            )
            self.num_pruned_samples += num_samples - len(selected_indices)

        self.numpy_rng.shuffle(selected_indices)
        return selected_indices

    @property
    def sampler(self):
        sampler = SADPSampler(self)
        if dist.is_available() and dist.is_initialized():
            sampler = DistributedSADPSampler(sampler)
        return sampler

    def no_prune(self):
        sample_indices = list(range(len(self)))
        self.numpy_rng.shuffle(sample_indices)
        return sample_indices

    def mean_score(self):
        return self.scores.mean()

    def get_weights(self, indexes):
        return self.weights[indexes]

    def get_pruned_count(self):
        return self.num_pruned_samples

    def reset_weights(self):
        self.weights.fill_(1.0)


class SADPSampler(object):
    def __init__(self, dataset: SADP):
        self.dataset = dataset
        self.iterations = 0
        self.sample_indices = None
        self.iter_obj = None
        self.reset()
        print("initialize sampler")

    def __getitem__(self, idx):
        return self.sample_indices[idx]

    def reset(self):
        if self.dataset.use_full_data:
            print(f"using full data... at iter{self.iterations}")
            self.dataset.reset_weights()
            self.sample_indices = self.dataset.no_prune()
            self.iter_obj = iter(self.sample_indices)
            return
        # print(f"reset at iter{self.iterations}")
        # seed = self.dataset.base_seed + self.iterations
        # self.dataset.numpy_rng = np.random.default_rng(seed)
        # self.dataset.bern_generator.manual_seed(seed)
        self.sample_indices = self.dataset.spike_aware_prune(
            iteration=self.iterations
        )
        self.iter_obj = iter(self.sample_indices)
        self.iterations += 1

    def __next__(self):
        return next(self.iter_obj)

    def __len__(self):
        return len(self.sample_indices)

    def __iter__(self):
        self.reset()
        return self


class DistributedSADPSampler(DistributedSampler):
    """Wrap SADPSampler for distributed training with a changing size."""

    class DatasetFromSampler(Dataset):
        def __init__(self, sampler: SADPSampler):
            self.dataset = sampler

        def reset(self):
            self.dataset.reset()

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, index: int):
            return self.dataset[index]

    def __init__(self,
                 dataset: SADPSampler,
                 num_replicas: Optional[int] = None,
                 rank: Optional[int] = None,
                 shuffle: bool = True,
                 seed: int = 0,
                 drop_last: bool = True) -> None:
        sampler = self.DatasetFromSampler(dataset)
        super(DistributedSADPSampler, self).__init__(
            sampler, num_replicas, rank, shuffle, seed, drop_last
        )
        self.sampler = sampler
        self.dataset = sampler.dataset.dataset
        self.iter_obj = None

    def __iter__(self) -> Iterator[int]:
        self.sampler.reset()
        if self.drop_last and len(self.sampler) % self.num_replicas != 0:
            self.num_samples = len(self.sampler) // self.num_replicas
        else:
            self.num_samples = math.ceil(
                len(self.sampler) / self.num_replicas
            )
        self.total_size = self.num_samples * self.num_replicas

        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(
                len(self.sampler), generator=generator
            ).tolist()
        else:
            indices = list(range(len(self.sampler)))

        if not self.drop_last:
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (
                    indices * math.ceil(padding_size / len(indices))
                )[:padding_size]
        else:
            indices = indices[:self.total_size]

        if len(indices) != self.total_size:
            raise RuntimeError('Distributed SADP sampler size mismatch')
        indices = indices[self.rank:self.total_size:self.num_replicas]

        # itemgetter(*indices) returns a scalar when a rank receives one item,
        # which is not iterable. Direct indexing handles every subset size.
        self.iter_obj = iter([self.sampler[index] for index in indices])
        return self.iter_obj
