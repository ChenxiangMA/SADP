import time
import argparse
import json
from datetime import datetime
import os
import logging


import torch
from torch.utils.tensorboard import SummaryWriter
from torch.cuda import amp

from load_dataset import load_dataset
from tools.utils import setup_logging, AverageMeter, ProgressMeter, accuracy
import models.spk_model
from tools import distributed



parser = argparse.ArgumentParser(description='PyTorch Training')
# args of datasets
parser.add_argument('--dataset', default='cifar10', type=str,
                    help='dataset: [cifar10|stl10|svhn]')
parser.add_argument('--data', default='/datasets/imagenet',
                    help='path to dataset, /datasets/imagenet,/data/public/imagenet2012')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--auto_aug', action='store_true', default=False, help='')
parser.add_argument('--cutout', action='store_true', default=False, help='')
# args of networks
parser.add_argument('--arch', default='spk_cifar_resnet', type=str)
parser.add_argument('--net', default='spk_resnet32', type=str, help='networks')
parser.add_argument('--seed', default=1234, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--use_ddp', action='store_true', default=False, help='')
parser.add_argument('--save_path', default='', type=str, help='the directory used to save the trained models')
parser.add_argument('--name', default='', type=str, help='')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=1024, type=int,
                    metavar='N',
                    help='mini-batch size')
parser.add_argument('-p', '--print_freq', default=100, type=int,
                    metavar='N', help='print frequency')

# args of optimizer
parser.add_argument('--optim', default='sgd', type=str, help='optimizer (default: sgd)')
parser.add_argument('--lr', '--learning_rate', default=0.4, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight_decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')

# args of spiking neural networks
parser.add_argument('--threshold', type=float, default=1., help='neuronal threshold (default: 1)')
parser.add_argument('--time_window', type=int, default=1, help='total time steps (default: 10)')
parser.add_argument('--decay', type=float, default=0.2, help='tau factor (default: 1)')
parser.add_argument('--pos_bn', type=int, default=3)
parser.add_argument('--dropout_rate', default=0.0, type=float, help='')


# =========== SADP BEGIN: spike-aware data pruning arguments ===========
parser.add_argument('--ratio', default=0.5, type=float, help='prune ratio')
parser.add_argument('--adaptive_prune_ratio', action='store_true', default=False)
parser.add_argument('--keep_ratio_gap_to_min', default=0.1, type=float)
parser.add_argument('--num_redistrib', type=int, default=6)
parser.add_argument('--smooth_constant', type=float, default=0.)
parser.add_argument('--ddp_rule', default='', type=str,
                    help='ddp rule: [sadp]')
parser.add_argument('--min_value', default=3e-5, type=float, help='')
# =========== SADP END: spike-aware data pruning arguments ===========

args = parser.parse_args()


def main():
    if args.save_path == '':
        save_path = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        save_path = save_path + args.name + '_' + str(args.seed)
    else:
        save_path = args.save_path

    if args.use_ddp:
        distributed.init_distributed_mode(args)

    if args.use_ddp:
        torch.distributed.barrier()
        if distributed.is_main_process():
            from pathlib import Path
            Path(save_path).mkdir(parents=True, exist_ok=True)
            # Logging settings
            setup_logging(os.path.join(save_path, 'log.txt'))
            logging.info('saving to:' + str(save_path))
    else:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # Logging settings
        setup_logging(os.path.join(save_path, 'log.txt'))
        logging.info('saving to:' + str(save_path))

    if args.use_ddp:
        is_cuda = torch.cuda.is_available()
        assert is_cuda
        device = torch.device('cuda', args.gpu)
        from tools import set_random_seed
        set_random_seed(seed=args.seed, is_ddp=True)
        bern_generator = torch.Generator(device=device)
        bern_generator.manual_seed(args.seed + 1)
        torch.backends.cudnn.benchmark = True
    else:
        is_cuda = torch.cuda.is_available()
        assert is_cuda, 'CPU is not supported!'
        device = torch.device('cuda' if is_cuda else 'cpu')
        from tools import set_random_seed
        set_random_seed(seed=args.seed, is_ddp=False)  # TODO
        bern_generator = torch.Generator(device=device)
        bern_generator.manual_seed(args.seed + 1)
        torch.backends.cudnn.benchmark = False
        args.gpu = 'cuda'

    # Load datasets
    train_loader, val_loader, num_classes = load_dataset(
        dataset=args.dataset, args=args,
        bern_generator=bern_generator, device=device
    )
    args.num_classes = num_classes

    args.save_path = save_path
    logging.info('args:' + str(args))

    # Load spiking model
    model = models.spk_model.SpkModel(args=args)

    # logging.info(str(model))

    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                                    momentum=args.momentum)
    elif args.optim == 'adam':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=args.epochs)
    scaler = amp.GradScaler()
    best_acc1 = 0

    if args.use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model.to(device),
            device_ids=[device],
            find_unused_parameters=True,
        )
    else:
        model = torch.nn.DataParallel(model).cuda()

    criterion = torch.nn.CrossEntropyLoss(
        # =========== SADP BEGIN: retain per-sample losses for weighting ===========
        reduction='none'
        # =========== SADP END: retain per-sample losses for weighting ===========
    )
    test_criterion = torch.nn.CrossEntropyLoss()

    standard_train(train_loader, val_loader, model, criterion, test_criterion, optimizer, scheduler, save_path,
                   best_acc1, scaler, args)




# =========== SADP BEGIN: final-layer scoring hooks ===========
hook_inputs = []
hook_gradients = []


def get_last_linear_layer(model):
    for name, layer in reversed(list(model.named_modules())):
        if isinstance(layer, torch.nn.Linear):
            return name, layer


def forward_hook(module, inputs, output):
    if not torch.is_grad_enabled():
        return

    x = inputs[0]
    with torch.no_grad():
        input_norm = torch.linalg.vector_norm(
            x.detach(),
            dim=tuple(range(1, x.ndim)),
            dtype=torch.float32,
        )

    hook_inputs.append(input_norm)


def backward_hook(module, grad_inputs, grad_outputs):
    grad = grad_outputs[0]
    if grad is not None:
        with torch.no_grad():
            gradient_norm = torch.linalg.vector_norm(
                grad.detach(),
                dim=tuple(range(1, grad.ndim)),
                dtype=torch.float32,
            )

        hook_gradients.append(gradient_norm)
# =========== SADP END: final-layer scoring hooks ===========



def standard_train(train_loader, val_loader, model, criterion, test_criterion, optimizer, scheduler, save_path,
                   best_acc1, scaler,
                   args):
    writer = SummaryWriter(save_path, purge_step=args.start_epoch)

    total_train_time = 0

    # =========== SADP BEGIN: scoring hook registration ===========
    if args.ddp_rule == 'sadp':
        layer_name, layer = get_last_linear_layer(model)
        layer.register_forward_hook(forward_hook)
        layer.register_full_backward_hook(backward_hook)
        logging.info(
            'Registered final Linear layer for importance scoring: %s (%s)',
            layer_name,
            layer,
        )
    # =========== SADP END: scoring hook registration ===========

    for epoch in range(args.start_epoch, args.epochs):

        if args.use_ddp:
            train_loader.sampler.set_epoch(epoch)


        logging.info(f"lr: {optimizer.param_groups[0]['lr']}")

        # train for one epoch
        train_start_time = time.time()

        train_acc1, train_loss = train_one_epoch(train_loader, model, criterion, optimizer, epoch, scaler,
                                                    args)
        total_train_time += time.time() - train_start_time

        
        logging.info(f"epoch:{epoch} training time: {total_train_time}")
        scheduler.step()
        if not args.use_ddp or (args.use_ddp and distributed.is_main_process()):
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
            writer.add_scalar('train_acc', train_acc1, epoch)
            writer.add_scalar('train_loss', train_loss, epoch)
            # evaluate on validation set
            val_acc1, val_loss = validate_one_epoch(val_loader, model, test_criterion, args)
            out_string = 'Train Acc. {:.4f} Test Acc. {:.4f} \t'.format(train_acc1,
                                                                        val_acc1,
                                                                        )
            writer.add_scalar('val_acc', val_acc1, epoch)
            writer.add_scalar('val_loss', val_loss, epoch)
            out_string += 'mem={:.0f}MiB, max_mem={:.0f}MiB, max_mem={:.0f}MiB \n'.format(
                torch.cuda.memory_allocated() / 1e6,
                torch.cuda.max_memory_allocated() / 1e6,
                torch.cuda.max_memory_reserved() / 1e6)
            logging.info(out_string)
            # remember best acc@1 and save checkpoint
            is_best = val_acc1 > best_acc1
            best_acc1 = max(val_acc1, best_acc1)

    if args.ddp_rule == 'sadp':
        logging.info(f"Total training time: {total_train_time}")
        logging.info(f'Total saved sample forwarding: {train_loader.dataset.get_pruned_count()}')
        
    logging.info(f'Best accuracy: {best_acc1.item()}')






# =========== SADP BEGIN: spike-aware importance score ===========
@torch.no_grad()
def get_spike_aware_score(input_norms, gradient_norms, loss_scale=1.0):
    """Compute per-sample scores, summing all time steps."""
    scores = torch.zeros_like(input_norms[0])

    for input_norm, gradient_norm in zip(input_norms, reversed(gradient_norms)):
        scores.addcmul_(input_norm, gradient_norm)

    return scores.div_(loss_scale).nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
# =========== SADP END: spike-aware importance score ===========





def train_one_epoch(train_loader, model, criterion, optimizer, epoch, scaler, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    # =========== SADP BEGIN: epoch for adaptive pruning ===========
    if args.adaptive_prune_ratio:
        train_loader.dataset.curr_epoch = epoch
    # =========== SADP END: epoch for adaptive pruning ===========
    iter_train_loader = iter(train_loader)
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    logging.info(f"Epoch: {epoch}, Iterations {len(train_loader)}")

    model.train()
    end = time.time()

    for i, (images, labels) in enumerate(iter_train_loader):
        data_time.update(time.time() - end)
        if args.dataset not in ['cifar10dvs', 'hardvs']:
            images = images.cuda(args.gpu, non_blocking=True)

        target = labels.cuda(args.gpu, non_blocking=True)
        
        # =========== SADP BEGIN: reset scoring caches before training forward ===========
        hook_inputs.clear()
        hook_gradients.clear()
        # =========== SADP END: reset scoring caches before training forward ===========
        optimizer.zero_grad(set_to_none=True)

        with amp.autocast():
            output = model(images)
            
            if args.dataset in ['imagenet', 'cifar10dvs', 'hardvs', 'cifar10', 'cifar100']:
                for each_time_step in range(output.size(1)):
                    if each_time_step == 0:
                        loss = criterion(output[:, each_time_step], target)
                    else:
                        loss = loss + criterion(output[:, each_time_step], target)
                loss = loss / output.size(1)
                # =========== SADP BEGIN: pruning loss weighting ===========
                if args.ddp_rule == 'sadp':
                    loss = train_loader.dataset.update(loss=loss)
                # =========== SADP END: pruning loss weighting ===========
                else:
                    loss = loss.mean()
                output = output.mean(1)
            else:
                raise NotImplementedError

        scaler.scale(loss).backward()

        # =========== SADP BEGIN: compute and update pruning scores ===========
        if args.ddp_rule == 'sadp':
            importance_scores = get_spike_aware_score(
                hook_inputs,
                hook_gradients,
                loss_scale=scaler.get_scale(),
            )
            train_loader.dataset.update_score(importance_scores)
            hook_inputs.clear()
            hook_gradients.clear()
        # =========== SADP END: compute and update pruning scores ===========

        scaler.step(optimizer)
        scaler.update()

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), target.size(0))

        top1.update(acc1[0], target.size(0))
        top5.update(acc5[0], target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        assert (i + 1) <= len(train_loader)
        if (i + 1) % args.print_freq == 0 or (i + 1) == len(train_loader):
            progress.display(i + 1)

    return top1.avg, losses.avg


def validate_one_epoch(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')


    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):

            if args.dataset not in ['cifar10dvs', 'hardvs']:
                images = images.cuda(args.gpu, non_blocking=True)
            else:
                assert target.size(0) == images[0].size(0)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output

            hook_inputs.clear()
            hook_gradients.clear()

            output = model(images)

            hook_inputs.clear()

            if args.dataset in ['imagenet', 'hardvs', 'cifar10dvs', 'cifar10', 'cifar100']:
                output = output.mean(1)

            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), target.size(0))
            top1.update(acc1[0], target.size(0))
            top5.update(acc5[0], target.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % args.print_freq == 0 or (i + 1) == len(val_loader):
                progress.display(i + 1)


    return top1.avg, losses.avg


if __name__ == '__main__':
    main()

