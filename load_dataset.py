import os

from torchvision import datasets, transforms
import torch
import logging

from datasets.autoaugment import CIFAR10Policy, Cutout

# =========== SADP BEGIN ===========
def get_data_pruned_dataset(train_dataset, args, bern_generator, device):
    if args.ddp_rule == 'sadp':
        from sadp import SADP
        logging.info('Use spike-aware data pruning')
        train_dataset = SADP(train_dataset,
                             num_epochs=args.epochs,
                             prune_ratio=args.ratio,
                             bern_generator=bern_generator,
                             adaptive_prune_ratio=args.adaptive_prune_ratio,
                             keep_ratio_gap_to_min=args.keep_ratio_gap_to_min,
                             num_redistrib=args.num_redistrib,
                             smooth_constant=args.smooth_constant,
                             min_value=args.min_value,
                             device=device)
    else:
        logging.info('No data pruning')
        train_dataset = train_dataset
    return train_dataset

def get_data_pruned_sampler(train_dataset, args):
    if args.ddp_rule == 'sadp':
        train_sampler = train_dataset.sampler
        if args.use_ddp:
            assert args.batch_size % args.world_size == 0
            batch_size = args.batch_size // args.world_size
        else:
            batch_size = args.batch_size
        train_shuffle = False
    elif args.use_ddp:
        train_sampler = torch.utils.data.DistributedSampler(
            train_dataset, shuffle=True)
        assert args.batch_size % args.world_size == 0
        batch_size = args.batch_size // args.world_size
        train_shuffle = False
    else:
        train_sampler = None
        batch_size = args.batch_size
        train_shuffle = True
    return train_sampler, batch_size, train_shuffle

# =========== SADP END ===========

def load_dataset(dataset='CIFAR10', args=None, bern_generator=None, device=None):
    if args.dataset in ['cifar10', 'cifar100']:
        args.image_size = 32
        if args.dataset == 'cifar10':
            num_classes = 10
            normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        elif args.dataset == 'cifar100':
            num_classes = 100
            normalize = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        train_augmentation = [transforms.RandomCrop(32, padding=4),
                              transforms.RandomHorizontalFlip()]
        if args.auto_aug:
            train_augmentation.append(CIFAR10Policy())
        train_augmentation.append(transforms.ToTensor())
        if args.cutout:
            train_augmentation.append(Cutout(n_holes=1, length=16))
        train_augmentation.append(normalize)
        train_transform = transforms.Compose(train_augmentation)
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

        train_dataset = datasets.__dict__[args.dataset.upper()]('../datasets/' + args.dataset, download=True,
                                                                transform=train_transform, train=True
                                                                )
        # =========== SADP BEGIN ===========
        train_dataset = get_data_pruned_dataset(train_dataset=train_dataset, args=args, bern_generator=bern_generator, device=device)
        train_sampler, batch_size, train_shuffle = get_data_pruned_sampler(train_dataset=train_dataset, args=args)
        # =========== SADP END ===========
        logging.info(f"train_sampler: {train_sampler}")
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   num_workers=args.workers,
                                                   shuffle=train_shuffle,
                                                   pin_memory=True,
                                                   sampler=train_sampler,
                                                   drop_last=True,
                                                   )
        val_loader = torch.utils.data.DataLoader(
            datasets.__dict__[args.dataset.upper()]('../datasets/' + args.dataset, transform=test_transform,
                                                    download=True,
                                                    train=False),
            batch_size=batch_size,
            num_workers=args.workers,
            pin_memory=True,
            shuffle=False)
    elif args.dataset == 'imagenet':
        num_classes = 1000
        traindir = os.path.join(args.data, 'train')
        valdir = os.path.join(args.data, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        train_dataset = datasets.ImageFolder(
            traindir,
            train_transform
        )
        args.image_size = 224
        # if args.use_ddp:
        #     train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
        #     assert args.batch_size % args.world_size == 0
        #     batch_size = args.batch_size // args.world_size
        # else:
        #     train_sampler = None
        #     batch_size = args.batch_size

        train_dataset = get_data_pruned_dataset(train_dataset=train_dataset, args=args, bern_generator=bern_generator,
                                                device=device)
        train_sampler, batch_size, train_shuffle = get_data_pruned_sampler(train_dataset=train_dataset, args=args)
        logging.info(f"train_sampler: {train_sampler}")
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   num_workers=args.workers,
                                                   shuffle=train_shuffle,
                                                   pin_memory=True,
                                                   sampler=train_sampler,
                                                   drop_last=True,
                                                   )
        # train_loader = torch.utils.data.DataLoader(
        #     train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
        #     num_workers=args.workers, pin_memory=True, sampler=train_sampler)

        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)

    elif args.dataset == 'hardvs':
        from datasets.hardvs.dvs_har import HARDVS
        from datasets.cifardvs.augmentation import RandomResizedCrop, ToPILImage, HorizontalFlip, ToTensor, Resize, CenterCrop, ToPILImageTensor

        transform_train = transforms.Compose(
            [
                ToPILImageTensor(),
                RandomResizedCrop(),
                HorizontalFlip(),
                ToTensor(),
            ]
        )
        transform_test = transforms.Compose(
            [ToPILImageTensor(), Resize(256), CenterCrop(224), ToTensor()]
        )
        data_path = '/datasets/hardvs'
        # data_path = '/home/chenxiangma/hardvs'
        train_dataset = HARDVS(root=data_path, train=True, data_type='frame', frames_number=args.time_window, split_by='number', transform=transform_train)

        train_dataset = get_data_pruned_dataset(train_dataset=train_dataset, args=args, bern_generator=bern_generator,
                                                device=device)
        train_sampler, batch_size, train_shuffle = get_data_pruned_sampler(train_dataset=train_dataset, args=args)
        logging.info(f"train_sampler: {train_sampler}")
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   num_workers=args.workers,
                                                   shuffle=train_shuffle,
                                                   pin_memory=True,
                                                   sampler=train_sampler,
                                                   drop_last=True,
                                                   )

        test_dataset = HARDVS(root=data_path, train=False, data_type='frame', frames_number=args.time_window, split_by='number', transform=transform_test)
        val_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                                           drop_last=False,
                                           pin_memory=True)

        args.image_size = 112
        num_classes = 300

    else:
        raise Exception('No valid dataset is specified.')
    return train_loader, val_loader, num_classes
