######################

# CIFAR10
CUDA_VISIBLE_DEVICES=0 python main_train.py --lr 0.2 --batch_size 128 --epochs 200 --pos_bn 3 --net spk_resnet18 --wd 5e-5 --decay 0.1 --threshold 1.0 --time_window 4 --cutout --auto_aug --ddp_rule sadp --ratio 0.3 --smooth_constant 0.35 --num_redistrib 6 --adaptive_prune_ratio --keep_ratio_gap_to_min 0.3 --min_value 0 --name cifar10_res18_sadp_03
CUDA_VISIBLE_DEVICES=0 python main_train.py --lr 0.2 --batch_size 128 --epochs 200 --pos_bn 3 --net spk_resnet18 --wd 5e-5 --decay 0.1 --threshold 1.0 --time_window 4 --cutout --auto_aug --ddp_rule sadp --ratio 0.5 --smooth_constant 0.3 --num_redistrib 6 --adaptive_prune_ratio --keep_ratio_gap_to_min 0.2 --name cifar10_res18_sadp_05
CUDA_VISIBLE_DEVICES=0 python main_train.py --lr 0.2 --batch_size 128 --epochs 200 --pos_bn 3 --net spk_resnet18 --wd 5e-5 --decay 0.1 --threshold 1.0 --time_window 4 --cutout --auto_aug --ddp_rule sadp --ratio 0.7 --smooth_constant 0.2 --num_redistrib 6 --adaptive_prune_ratio --keep_ratio_gap_to_min 0.2 --name cifar10_res18_sadp_07

#CIFAR100
CUDA_VISIBLE_DEVICES=0 python main_train.py --dataset cifar100 --lr 0.2 --batch_size 128 --epochs 200 --pos_bn 3 --net spk_resnet19 --wd 5e-5 --decay 0.1 --threshold 1.0 --time_window 4 --cutout --auto_aug --ddp_rule sadp --ratio 0.3 --smooth_constant 0.35 --num_redistrib 6 --adaptive_prune_ratio --keep_ratio_gap_to_min 0.3 --min_value 0 --name cifar100_res19_sadp_03
CUDA_VISIBLE_DEVICES=0 python main_train.py --dataset cifar100 --lr 0.2 --batch_size 128 --epochs 200 --pos_bn 3 --net spk_resnet19 --wd 5e-5 --decay 0.1 --threshold 1.0 --time_window 4 --cutout --auto_aug --ddp_rule sadp --ratio 0.5 --smooth_constant 0.3 --num_redistrib 6 --adaptive_prune_ratio --keep_ratio_gap_to_min 0.2 --name cifar100_res19_sadp_05
CUDA_VISIBLE_DEVICES=0 python main_train.py --dataset cifar100 --lr 0.2 --batch_size 128 --epochs 200 --pos_bn 3 --net spk_resnet19 --wd 5e-5 --decay 0.1 --threshold 1.0 --time_window 4 --cutout --auto_aug --ddp_rule sadp --ratio 0.7 --smooth_constant 0.2 --num_redistrib 6 --adaptive_prune_ratio --keep_ratio_gap_to_min 0.2 --name cifar100_res19_sadp_07

# HAR-DVS
#CUDA_VISIBLE_DEVICES=0 python main_train.py --dataset hardvs   --lr 1e-3 --optim adam --batch_size 100 --epochs 100 --pos_bn 3 --arch spk_cifar_resnet --net spk_resnet18 --wd 5e-4 --decay 0.1 --threshold 1.0 --time_window 4 -j 16 --ddp_rule sadp --ratio 0.3 --smooth_constant 0.35 --num_redistrib 6 --adaptive_prune_ratio --keep_ratio_gap_to_min 0.3 --min_value 0 --name HARDVS_res18_sadp_03
#CUDA_VISIBLE_DEVICES=0 python main_train.py --dataset hardvs   --lr 1e-3 --optim adam --batch_size 100 --epochs 100 --pos_bn 3 --arch spk_cifar_resnet --net spk_resnet18 --wd 5e-4 --decay 0.1 --threshold 1.0 --time_window 4 -j 16 --ddp_rule sadp --ratio 0.5 --smooth_constant 0.3 --num_redistrib 6 --adaptive_prune_ratio --keep_ratio_gap_to_min 0.2 --name HARDVS_res18_sadp_05
#CUDA_VISIBLE_DEVICES=0 python main_train.py --dataset hardvs   --lr 1e-3 --optim adam --batch_size 100 --epochs 100 --pos_bn 3 --arch spk_cifar_resnet --net spk_resnet18 --wd 5e-4 --decay 0.1 --threshold 1.0 --time_window 4 -j 16 --ddp_rule sadp --ratio 0.7 --smooth_constant 0.2 --num_redistrib 6 --adaptive_prune_ratio --keep_ratio_gap_to_min 0.2 --name HARDVS_res18_sadp_07











