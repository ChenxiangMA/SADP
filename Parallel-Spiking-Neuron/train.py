import datetime
import os
import sys
import time

_PROGRAM_START_TIME = time.perf_counter()

# Keep shared project modules importable when running from this directory.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(1, _PROJECT_ROOT)

import torch
import torch.utils.data
from torch import nn
import torchvision
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import math
from torch.cuda import amp
import torch.distributed.optim
import argparse
import torch.nn.functional as F
from spikingjelly.activation_based import functional, surrogate
import sew_resnet, utils


def temporal_efficient_training_cross_entropy(x_seq, target):
    """
    :param x_seq: ``shape=[T, N, C, *]``, where ``C`` is the number of classes
    :type x_seq: Tensor
    :param target: ``shape=[N]``, where ``0 <= target[i] <= C-1``
    :type target: torch.LongTensor
    :return: the temporal efficient training cross entropy
    :rtype: Tensor

    The temporal efficient training (TET) cross entropy, which is the mean of cross entropy of each time-step.

    Codes example:

    .. code-block:: python

        def tet_ce_for_loop_version(x_seq: Tensor, target: torch.LongTensor):
            loss = 0.
            for t in range(x_seq.shape[0]):
                loss += F.cross_entropy(x_seq[t], target)
            return loss / x_seq.shape[0]

        T = 8
        N = 4
        C = 10
        x_seq = torch.rand([T, N, C])
        target = torch.randint(low=0, high=C-1, size=[N])
        print(tet_ce_for_loop_version(x_seq, target))
        print(temporal_efficient_training_cross_entropy(x_seq, target))


    .. admonition:: Tip
        :class: tip

        The TET cross entropy is proposed by `Temporal Efficient Training of Spiking Neural Network via Gradient Re-weighting <https://openreview.net/forum?id=_XNtisL32jv>`_.
    """
    x_seq = x_seq.transpose(0, 1).transpose(1, 2)  # [N, C, T, *]
    N, C, T = x_seq.shape[0], x_seq.shape[1], x_seq.shape[2]
    if x_seq.dim() == 3:
        # x_seq.shape = [N, C, T]
        # target.shape = [N]
        target = target.unsqueeze(1).repeat(1, T)  # [N, T]
    else:
        # x_seq.shape = [N, C, T, d1, d2, ..., dk]
        # target.shape = [N, d1, d2, ..., dk]
        rep_shape = [1, T]
        rep_shape.extend([1] * (x_seq.dim() - 3))
        target = target.unsqueeze(1).repeat(rep_shape)

    loss = F.cross_entropy(x_seq, target)
    return loss


class IFNode5(nn.Module):
    def __init__(self, T: int, surrogate_function: surrogate.SurrogateFunctionBase):
        super().__init__()
        self.surrogate_function = surrogate_function
        self.fc = nn.Linear(T, T)
        nn.init.constant_(self.fc.bias, -1)

    def forward(self, x_seq: torch.Tensor):
        # x_seq.shape = [T, N, *]
        h_seq = torch.addmm(self.fc.bias.unsqueeze(1), self.fc.weight, x_seq.flatten(1))
        spike = self.surrogate_function(h_seq)
        return spike.view(x_seq.shape)


_seed_ = 2025
import random
random.seed(2025)

torch.manual_seed(_seed_)  # use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA)
torch.cuda.manual_seed_all(_seed_)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import numpy as np
np.random.seed(_seed_)

import numpy as np
np.random.seed(_seed_)



hook_outputs = list()
hook_gradients = list()


def hook_fn(module, module_in, module_out):
    if not torch.is_grad_enabled():
        return

    with torch.no_grad():
        input_norm = torch.linalg.vector_norm(
            module_in[0].detach(),
            dim=-1,
            dtype=torch.float32,
        )
    hook_outputs.append(input_norm)


def backward_hook(module, gradients_in, gradients_out):
    gradient = gradients_out[0]
    if gradient is None:
        return

    with torch.no_grad():
        gradient_norm = torch.linalg.vector_norm(
            gradient.detach(),
            dim=-1,
            dtype=torch.float32,
        )
    hook_gradients.append(gradient_norm)


@torch.no_grad()
def get_spike_aware_score(
        layer_wise_outputs,
        layer_wise_gradients,
        loss_scale=1.0):
    if len(layer_wise_outputs) != len(layer_wise_gradients):
        raise RuntimeError(
            'Captured input and gradient counts do not match'
        )
    if len(layer_wise_outputs) != 1:
        raise RuntimeError(
            'PSN currently expects exactly one hooked FC layer'
        )
    if loss_scale <= 0:
        raise ValueError(
            f'loss_scale must be positive, got {loss_scale}'
        )

    score = (
        layer_wise_outputs[0] * layer_wise_gradients[0]
    ).sum(dim=0)

    score.div_(loss_scale)
    score.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
    return score



def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, print_freq, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('img/s', utils.SmoothedValue(window_size=10, fmt='{value}'))

    header = 'Epoch: [{}]'.format(epoch)

    if args.ddp_rule == 'sadp':
        data_loader.dataset.curr_epoch = epoch

    global hook_outputs
    global hook_gradients

    iter_train_loader = iter(data_loader)
    torch.distributed.barrier()
    for image, target in metric_logger.log_every(data_loader, print_freq, header, iter_train_loader=iter_train_loader):
        start_time = time.time()
        image, target = image.to(device), target.to(device)
        hook_outputs = list()
        hook_gradients = list()
        # with torch.autograd.detect_anomaly():
        if scaler is not None:
            with amp.autocast():
                output = model(image)
                loss = 0.
                for t in range(output.size(0)):
                    loss = loss + criterion(output[t], target)
                loss = loss / output.size(0)
                if args.ddp_rule == 'sadp':
                    loss = data_loader.dataset.update(loss=loss)
                else:
                    loss = loss.mean()
                optimizer.zero_grad()
                scaler.scale(loss).backward()

                if args.ddp_rule == 'sadp':
                    with torch.no_grad():
                        importance_scores = get_spike_aware_score(
                            hook_outputs,
                            hook_gradients,
                            loss_scale=scaler.get_scale(),
                        )
                        data_loader.dataset.update_score(importance_scores)
                scaler.step(optimizer)
                scaler.update()
        else:
            raise NotImplementedError

        functional.reset_net(model)
        if output.dim() == 3:
            with torch.no_grad():
                output = output.mean(0)

        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        batch_size = image.shape[0]
        loss_s = loss.item()
        if math.isnan(loss_s):
            raise ValueError('loss is Nan')
        acc1_s = acc1.item()
        acc5_s = acc5.item()

        metric_logger.update(loss=loss_s, lr=optimizer.param_groups[0]["lr"])

        metric_logger.meters['acc1'].update(acc1_s, n=batch_size)
        metric_logger.meters['acc5'].update(acc5_s, n=batch_size)
        metric_logger.meters['img/s'].update(batch_size / (time.time() - start_time))

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    return metric_logger.loss.global_avg, metric_logger.acc1.global_avg, metric_logger.acc5.global_avg



def evaluate(model, criterion, data_loader, device, print_freq=100, header='Test:'):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    global hook_outputs
    global hook_gradients
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, print_freq, header, iter(data_loader)):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            hook_outputs = list()
            hook_gradients = list()
            output = model(image)
            output = output.mean(0)
            loss = criterion(output, target)
            functional.reset_net(model)

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    loss, acc1, acc5 = metric_logger.loss.global_avg, metric_logger.acc1.global_avg, metric_logger.acc5.global_avg
    print(f' * Acc@1 = {acc1}, Acc@5 = {acc5}, loss = {loss}')
    return loss, acc1, acc5


def _get_cache_path(filepath):
    import hashlib
    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join("~", ".torch", "vision", "datasets", "imagefolder", h[:10] + ".pt")
    cache_path = os.path.expanduser(cache_path)
    return cache_path

def load_data(traindir, valdir, cache_dataset, distributed):
    # Data loading code
    print("Loading data")
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    print("Loading training data")
    st = time.time()
    cache_path = _get_cache_path(traindir)
    if cache_dataset and os.path.exists(cache_path):
        # Attention, as the transforms are also cached!
        print("Loading dataset_train from {}".format(cache_path))
        dataset, _ = torch.load(cache_path)
    else:
        dataset = torchvision.datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
        if cache_dataset:
            print("Saving dataset_train to {}".format(cache_path))
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((dataset, traindir), cache_path)
    print("Took", time.time() - st)

    print("Loading validation data")
    cache_path = _get_cache_path(valdir)
    if cache_dataset and os.path.exists(cache_path):
        # Attention, as the transforms are also cached!
        print("Loading dataset_test from {}".format(cache_path))
        dataset_test, _ = torch.load(cache_path)
    else:
        dataset_test = torchvision.datasets.ImageFolder(
            valdir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))
        if cache_dataset:
            print("Saving dataset_test to {}".format(cache_path))
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((dataset_test, valdir), cache_path)

    print("Creating data loaders")
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return dataset, dataset_test, train_sampler, test_sampler


def main(args):
    max_test_acc1 = 0.
    test_acc5_at_max_test_acc1 = 0.


    train_tb_writer = None
    te_tb_writer = None


    utils.init_distributed_mode(args)
    print(args)
    output_dir = os.path.join(args.output_dir, f'{args.model}_b{args.batch_size}_lr{args.lr}_T{args.T}')

    if args.weight_decay:
        output_dir += f'_wd{args.weight_decay}'

    if args.cos_lr_T == -1:
        args.cos_lr_T = args.epochs

    output_dir += f'_coslr{args.cos_lr_T}'

    if args.adamw:
        output_dir += '_adamw'
    else:
        output_dir += '_sgd'

    output_dir += f'_{args.world_size}gpu'

    if args.load is not None:
        output_dir += '_load'

    if args.tet:
        output_dir += '_tet'

    output_dir += str(args.ddp_rule)
    output_dir += '2'

    if output_dir:
        utils.mkdir(output_dir)


    device = torch.device(args.device)

    train_dir = os.path.join(args.data_path, 'train')
    val_dir = os.path.join(args.data_path, 'val')
    dataset_train, dataset_test, train_sampler, test_sampler = load_data(train_dir, val_dir,
                                                                   args.cache_dataset, args.distributed)


    if args.ddp_rule == 'sadp':
        from sadp import SADP
        print('Use SADP.')
        sadp_device = (
            torch.device('cuda', args.gpu)
            if args.distributed
            else device
        )

        bern_generator = torch.Generator(device=sadp_device)
        bern_generator.manual_seed(2025)

        dataset_train = SADP(
            dataset=dataset_train,
            num_epochs=args.epochs,
            prune_ratio=args.ratio,
            bern_generator=bern_generator,
            adaptive_prune_ratio=args.adaptive_prune_ratio,
            keep_ratio_gap_to_min=args.keep_ratio_gap_to_min,
            num_redistrib=args.num_redistrib,
            smooth_constant=args.smooth_constant,
            min_value=args.min_value,
            device=sadp_device,
        )
        train_sampler = dataset_train.sampler

    data_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.workers, pin_memory=True, drop_last=True)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=100,
        sampler=test_sampler, num_workers=args.workers, pin_memory=True)

    print("Creating model")

    if args.model in sew_resnet.__dict__:
        model = sew_resnet.__dict__[args.model](pretrained=False, cnf='ADD', spiking_neuron=IFNode5,
                                                surrogate_function=surrogate.ATan(), T=args.T)
    else:
        raise NotImplementedError(args.model)


    # print(model)

    if args.load is not None:
        model.load_state_dict(torch.load(args.load), strict=False)
        print('load', args.load)


    model.to(device)
    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    criterion = nn.CrossEntropyLoss(reduction='none')
    test_criterion = nn.CrossEntropyLoss()



    if args.adamw:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    if args.amp:
        scaler = amp.GradScaler()
    else:
        scaler = None

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.cos_lr_T)



    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

        args.start_epoch = checkpoint['epoch'] + 1

        max_test_acc1 = checkpoint['max_test_acc1']
        test_acc5_at_max_test_acc1 = checkpoint['test_acc5_at_max_test_acc1']

    if args.test_only:
        evaluate(model, criterion, data_loader_test, device=device, header='Test:')
        return

    if args.tb and utils.is_main_process():
        purge_step_train = args.start_epoch
        purge_step_te = args.start_epoch
        train_tb_writer = SummaryWriter(output_dir + '_logs/train', purge_step=purge_step_train)
        te_tb_writer = SummaryWriter(output_dir + '_logs/te', purge_step=purge_step_te)
        with open(output_dir + '_logs/args.txt', 'w', encoding='utf-8') as args_txt:
            args_txt.write(str(args))

        print(f'purge_step_train={purge_step_train}, purge_step_te={purge_step_te}')

    print("Start training")
    total_train_time = 0


    if args.ddp_rule == 'sadp':
        for name, layer in model.module.named_modules():
            if name == 'fc':
                print(f"layer name: {name}")
                layer.register_forward_hook(hook_fn)
                layer.register_full_backward_hook(backward_hook)

    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        save_max = False
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_start_time = time.time()
        train_loss, train_acc1, train_acc5 = train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, args.print_freq, scaler)
        total_train_time += time.time() - train_start_time
        if utils.is_main_process():
            train_tb_writer.add_scalar('train_loss', train_loss, epoch)
            train_tb_writer.add_scalar('train_acc1', train_acc1, epoch)
            train_tb_writer.add_scalar('train_acc5', train_acc5, epoch)
        lr_scheduler.step()

        test_loss, test_acc1, test_acc5 = evaluate(model, test_criterion, data_loader_test, device=device, header='Test:')
        if te_tb_writer is not None:
            if utils.is_main_process():

                te_tb_writer.add_scalar('test_loss', test_loss, epoch)
                te_tb_writer.add_scalar('test_acc1', test_acc1, epoch)
                te_tb_writer.add_scalar('test_acc5', test_acc5, epoch)


        if max_test_acc1 < test_acc1:
            max_test_acc1 = test_acc1
            test_acc5_at_max_test_acc1 = test_acc5
            save_max = True



        if output_dir:

            checkpoint = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
                'max_test_acc1': max_test_acc1,
                'test_acc5_at_max_test_acc1': test_acc5_at_max_test_acc1,
            }

            utils.save_on_master(
                checkpoint,
                os.path.join(output_dir, 'checkpoint_latest.pth'))
            save_flag = False

            if epoch % 64 == 0 or epoch == args.epochs - 1:
                save_flag = True

            elif args.cos_lr_T == 0:
                for item in args.lr_step_size:
                    if (epoch + 2) % item == 0:
                        save_flag = True
                        break

            if save_flag:
                utils.save_on_master(
                    checkpoint,
                    os.path.join(output_dir, f'checkpoint_{epoch}.pth'))

            if save_max:
                utils.save_on_master(
                    checkpoint,
                    os.path.join(output_dir, 'checkpoint_max_test_acc1.pth'))
        print(args)
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(output_dir)

        print('Training time {}'.format(total_time_str), 'max_test_acc1', max_test_acc1,
              'test_acc5_at_max_test_acc1', test_acc5_at_max_test_acc1)
        print(f"Total training time: {total_train_time}")
        if args.ddp_rule == 'sadp':
            print(f'Total saved sample forwarding: {data_loader.dataset.get_pruned_count()}')

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Classification Training')

    parser.add_argument('--data-path', default='/home/wfang/datasets/ImageNet', help='dataset')

    parser.add_argument('--model', default='resnet18', help='model')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=32, type=int)
    parser.add_argument('--epochs', default=320, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--lr', default=0.0025, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='Momentum for SGD. Adam will not use momentum')
    parser.add_argument('--wd', '--weight-decay', default=0, type=float,
                        metavar='W', help='weight decay (default: 0)',
                        dest='weight_decay')
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='.', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument(
        "--cache-dataset",
        dest="cache_dataset",
        help="Cache the datasets for quicker initialization. It also serializes the transforms",
        action="store_true",
    )
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )

    # Mixed precision training parameters
    parser.add_argument('--amp', action='store_true',
                        help='Use AMP training')


    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')


    parser.add_argument('--tb', action='store_true',
                        help='Use TensorBoard to record logs')
    parser.add_argument('--T', default=4, type=int, help='simulation steps')
    parser.add_argument('--adamw', action='store_true',
                        help='Use AdamW. The default optimizer is SGD.')

    parser.add_argument('--cos_lr_T', default=-1, type=int,
                        help='T_max of CosineAnnealingLR.')


    parser.add_argument('--load', type=str, default=None, help='the pt file path for loading pre-trained ANN weights')
    parser.add_argument('--tet', action='store_true', help='use the tet loss')

    ################################### Data Pruning Args #########################
    parser.add_argument('--ratio', default=0.5, type=float, help='prune ratio')
    parser.add_argument('--ddp_rule', default='', type=str,
                        help='ddp rule: [sadp|random|el2n|forgetting]')
    parser.add_argument('--smooth_constant', type=float, default=0.)
    parser.add_argument('--num_redistrib', type=int, default=6)
    parser.add_argument('--adaptive_prune_ratio', action='store_true', default=False,
                        help='whether use lr scaling or not.')
    parser.add_argument('--keep_ratio_gap_to_min', default=0.1, type=float)
    parser.add_argument('--min_value', type=float, default=0.0, help='Scores no greater than this value are treated as zero.',)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)


