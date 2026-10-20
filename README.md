# SADP
This repo contains the PyTorch implementation of the paper: [Efficient Training of Spiking Neural Networks by Spike-aware Data Pruning](https://arxiv.org/abs/2510.04098).

Additional code required for SADP is marked with `===` comments in `main_train.py`.

## Usage
### Dependencies
- pytorch

### Scripts

- Train ResNet18 with SADP on CIFAR10 at 30% pruning ratio:
```bash
python main_train.py --lr 0.2 --batch_size 128 --epochs 200 --pos_bn 3 --net spk_resnet18 --wd 5e-5 --decay 0.1 --threshold 1.0 --time_window 4 --cutout --auto_aug --ddp_rule sadp --ratio 0.3 --smooth_constant 0.35 --num_redistrib 6 --adaptive_prune_ratio --keep_ratio_gap_to_min 0.3 --min_value 0 --name cifar10_res18_sadp_03 
```

- Train ResNet19 with SADP on CIFAR100 at 50% pruning ratio:

```bash
python main_train.py --dataset cifar100 --lr 0.2 --batch_size 128 --epochs 200 --pos_bn 3 --net spk_resnet19 --wd 5e-5 --decay 0.1 --threshold 1.0 --time_window 4 --cutout --auto_aug --ddp_rule sadp --ratio 0.5 --smooth_constant 0.3 --num_redistrib 6 --adaptive_prune_ratio --keep_ratio_gap_to_min 0.2 --name cifar100_res19_sadp_05 
```

See `run.sh` for training commands at other pruning ratios.


## Contact
If you have any questions, please contact <chenxiang.ma@connect.polyu.hk>. 
