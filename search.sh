# CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7,0 nohup python -m torch.distributed.launch search.py \
# --ngpus_per_node 8 \
# --port 4568 \
# --distributed True \
# --dataset imagenet \
# --search_space AddShift \
# --dataset_path /Datadisk/shared-corpus/imagenet-mxnet/imagenet \
# --pretrain_epoch 60 \
# --act_num 2 \
# --batch_size 100 \
# --header_channel 1204 \
# --flops_weight 1e-10 \
# --efficiency_metric flops \
# --flops_max 2.5e8 \
# --flops_min 2.5e8 \
# --pretrain /Datadisk/shihh/NAS/ImageNet/AddShift_60_2_1 \
# --gpu 0,1,2,3,4,5,6,7 > ImageNet/AddShift_60_2_1 2>&1 &


# gpu=(3 4 5 6)
# space=(AddShift AddShift NoConv AddShift)
# bn=(128 128 128 128)
# for i in 0 1 2
# do
#     CUDA_VISIBLE_DEVICES=${gpu[$i]} nohup python -u search.py \
#     --dataset cifar10 \
#     --search_space ${space[$i]} \
#     --dataset_path /Datadisk/datasets/cifar10 \
#     --batch_size ${bn[$i]} \
#     --pretrain ckpt/${space[$i]} \
#     --seed 2 \
#     --gpu ${gpu[$i]} > ${space[$i]} 2>&1 &
#     done

# CUDA_VISIBLE_DEVICES=1 python -u search.py --dataset cifar10 --search_space AddShift --dataset_path /media/HardDisk1/cifar/cifar10 --batch_size 128 --pretrain ckpt/AddShift_130 --gpu 1

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
# --dataset cifar10 \
# --search_space AddShift \
# --dataset_path /Datadisk/datasets/cifar10 \
# --pretrain_epoch 130 \
# --act_num 3 \
# --lr 0.05 \
# --flops_weight 1e-10 \
# --efficiency_metric flops \
# --batch_size 128 \
# --pretrain /Datadisk/shihh/NAS/hw/cifar10_AddShift3 \
# --gpu 0 > hw/cifar10_AddShift2_130 2>&1 &

# CUDA_VISIBLE_DEVICES=0 nohup python -u search.py \
# --ngpus_per_node 2 \
# --port 4657 \
# --dataset cifar100 \
# --search_space AddAdd \
# --dataset_path /Datadisk/datasets/CIFAR100 \
# --load_epoch 60 \
# --pretrain_epoch 120 \
# --act_num 2 \
# --flops_weight 1e-10 \
# --efficiency_metric flops \
# --flops_max 1e8 \
# --flops_min 5e7 \
# --lr 0.05 \
# --lr_schedule cosine \
# --weight_decay 5e-4 \
# --arch_update_frec 1 \
# --batch_size 128 \
# --pretrain /Datadisk/shihh/NAS/ckpt/cifar100_AddAdd_pure_add_2 \
# --gpu 0 > pretrain/cifar100_AddAdd_pure_add_2 2>&1 &

# fix_conv:conv_allchannl,add_allchannel,mix,search
# fix_conv2:conv_allchannl,conv_all,add_allchannel,add_all,mix,search
# fix_conv3:conv_allchannl,conv_all,add_allchannel,add_all,search
# fix_conv4:conv_allchannl,conv_all,add_allchannel,add_all,search(conv_nograd)

CUDA_VISIBLE_DEVICES=2 nohup python -u search.py \
--ngpus_per_node 2 \
--port 4657 \
--dataset cifar10 \
--search_space AddAll \
--dataset_path /media/HD1/cifar/CIFAR10 \
--load_epoch 120 \
--pretrain_epoch 120 \
--act_num 2 \
--flops_weight 1.2e-10 \
--efficiency_metric flops \
--flops_max 1e8 \
--flops_min 5e7 \
--lr 0.1 \
--lr_schedule cosine \
--weight_decay 1e-4 \
--arch_update_frec 1 \
--batch_size 128 \
--pretrain /media/HD1/shh/NAS/ckpt/cifar10_AddAll_120_1.5_1.2_2 \
--gpu 0 > AddAll_CIFAR10/cifar10_AddAll_120_1.5_1.2_2 2>&1 &