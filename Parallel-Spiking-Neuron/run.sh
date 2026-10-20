CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --cos_lr_T 100 --model sew_resnet18 --output-dir ./logs --tb --print-freq 4096 --amp --T 4 --lr 0.3 --epoch 100 --data-path /datasets/imagenet --load ./resnet18-f37072fd.pth -b 256  --ratio 0.35 --ddp_rule sadp --smooth_constant 0.325 --num_redistrib 6  --adaptive_prune_ratio --keep_ratio_gap_to_min 0.2




