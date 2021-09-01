# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup python -m torch.distributed.launch search.py \
# --ngpus_per_node 8 \
# --port 4568 \
# --distributed True \
# --dataset imagenet \
# --search_space OnlyConv \
# --dataset_path /media/HardDisk1/jmlu/imagenet-mxnet/imagenet \
# --pretrain_epoch 30 \
# --act_num 3 \
# --batch_size 192 \
# --header_channel 1504 \
# --flops_weight 1e-10 \
# --efficiency_metric flops \
# --flops_max  \
# --flops_min  \
# --pretrain /media/HardDisk1/shihh/NAS/ckpt/imagenet_OnlyConv \
# --gpu 0,1,2,3,4,5,6,7 > imagenet_OnlyConv 2>&1 &


# gpu=(3 4 5 6)
# space=(OnlyConv OnlyConv NoConv OnlyConv)
# bn=(128 128 128 128)
# for i in 0 1 2
# do
#     CUDA_VISIBLE_DEVICES=${gpu[$i]} nohup python -u search.py \
#     --dataset cifar100 \
#     --search_space ${space[$i]} \
#     --dataset_path /Datadisk/datasets/CIFAR100 \
#     --batch_size ${bn[$i]} \
#     --pretrain ckpt/${space[$i]} \
#     --seed 2 \
#     --gpu ${gpu[$i]} > ${space[$i]} 2>&1 &
#     done

# CUDA_VISIBLE_DEVICES=1 python -u search.py --dataset cifar100 --search_space OnlyConv --dataset_path /media/HardDisk1/cifar/CIFAR100 --batch_size 128 --pretrain ckpt/OnlyConv_130 --gpu 1

# CUDA_VISIBLE_DEVICES=1,2,3,4,6,7 nohup python -m torch.distributed.launch search.py \
# --ngpus_per_node 6 \
# --distributed True \
# --port 4876 \
# --dataset imagenet \
# --search_space OnlyConv \
# --dataset_path /media/HardDisk1/jmlu/imagenet-mxnet/imagenet \
# --pretrain_epoch 30 \
# --act_num 3 \
# --lr 0.05 \
# --batch_size 128 \
# --pretrain /media/HardDisk1/shihh/NAS/ckpt/Imagenet_OnlyConv \
# --gpu 0,1,2,3,4,5 > Imagenet_OnlyConv_30 2>&1 &

# CUDA_VISIBLE_DEVICES=7 nohup python -u search.py \
# --ngpus_per_node 2 \
# --port 4657 \
# --dataset cifar100 \
# --search_space OnlyConv \
# --dataset_path /Datadisk/datasets/CIFAR100 \
# --pretrain_epoch 130 \
# --act_num 3 \
# --lr 0.05 \
# --flops_weight 1e-10 \
# --efficiency_metric flops \
# --batch_size 128 \
# --pretrain /Datadisk/shihh/NAS/hw/CIFAR100_OnlyConv3 \
# --gpu 0 > hw/CIFAR100_OnlyConv3_130 2>&1 &

CUDA_VISIBLE_DEVICES=4 nohup python -u search.py \
--ngpus_per_node 2 \
--port 4657 \
--dataset cifar100 \
--search_space OnlyConv \
--dataset_path /Datadisk/datasets/CIFAR100 \
--pretrain_epoch 30 \
--act_num 2 \
--lr 0.05 \
--batch_size 128 \
--pretrain /Datadisk/shihh/NAS/baseline/CIFAR100_OnlyConv \
--gpu 0 > baseline/CIFAR100_OnlyConv_2 2>&1 &