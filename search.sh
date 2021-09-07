# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup python -m torch.distributed.launch search.py \
# --ngpus_per_node 8 \
# --port 4568 \
# --distributed True \
# --dataset imagenet \
# --search_space AddShift \
# --dataset_path /media/HardDisk1/jmlu/imagenet-mxnet/imagenet \
# --pretrain_epoch 30 \
# --act_num 3 \
# --batch_size 192 \
# --header_channel 1504 \
# --flops_weight 1e-10 \
# --efficiency_metric flops \
# --flops_max  \
# --flops_min  \
# --pretrain /media/HardDisk1/shihh/NAS/ckpt/imagenet_AddShift \
# --gpu 0,1,2,3,4,5,6,7 > imagenet_AddShift 2>&1 &


# gpu=(3 4 5 6)
# space=(AddShift AddShift NoConv AddShift)
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

# CUDA_VISIBLE_DEVICES=1 python -u search.py --dataset cifar100 --search_space AddShift --dataset_path /media/HardDisk1/cifar/CIFAR100 --batch_size 128 --pretrain ckpt/AddShift_130 --gpu 1

# CUDA_VISIBLE_DEVICES=1,2,3,4,6,7 nohup python -m torch.distributed.launch search.py \
# --ngpus_per_node 6 \
# --distributed True \
# --port 4876 \
# --dataset imagenet \
# --search_space AddShift \
# --dataset_path /media/HardDisk1/jmlu/imagenet-mxnet/imagenet \
# --pretrain_epoch 30 \
# --act_num 3 \
# --lr 0.05 \
# --batch_size 128 \
# --pretrain /media/HardDisk1/shihh/NAS/ckpt/Imagenet_AddShift \
# --gpu 0,1,2,3,4,5 > Imagenet_AddShift_30 2>&1 &

# CUDA_VISIBLE_DEVICES=7 nohup python -u search.py \
# --ngpus_per_node 2 \
# --port 4657 \
# --dataset cifar100 \
# --search_space AddShift \
# --dataset_path /Datadisk/datasets/CIFAR100 \
# --pretrain_epoch 130 \
# --act_num 3 \
# --lr 0.05 \
# --flops_weight 1e-10 \
# --efficiency_metric flops \
# --batch_size 128 \
# --pretrain /Datadisk/shihh/NAS/hw/CIFAR100_AddShift3 \
# --gpu 0 > hw/CIFAR100_AddShift3_130 2>&1 &

CUDA_VISIBLE_DEVICES=6 nohup python -u search.py \
--ngpus_per_node 2 \
--port 4657 \
--dataset cifar100 \
--search_space AddShift \
--dataset_path /Datadisk/datasets/CIFAR100 \
--pretrain_epoch 40 \
--act_num 2 \
--flops_weight 0.5e-10 \
--efficiency_metric flops \
--flops_max 1e8 \
--flops_min 5e7 \
--lr 0.05 \
--batch_size 128 \
--pretrain /Datadisk/shihh/NAS/baseline/CIFAR100_AddShift_2_0.5e-10 \
--gpu 0 > baseline/CIFAR100_AddShift_2_0.5e-10 2>&1 &