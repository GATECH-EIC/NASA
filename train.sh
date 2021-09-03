CUDA_VISIBLE_DEVICES=7 nohup python -u train.py \
--ngpus_per_node 8 \
--dataset cifar10 \
--search_space OnlyConv \
--dataset_path /Datadisk/datasets/CIFAR10 \
--pretrain /Datadisk/shihh/NAS/baseline/CIFAR10_OnlyConv_2_scratch \
--nepochs 650 \
--batch_size 196 \
--lr 0.02 \
--load_path /Datadisk/shihh/NAS/baseline/CIFAR10_OnlyConv_2 \
--load_epoch 120 \
--gpu 0 > baseline/CIFAR10_OnlyConv_2_scratch 2>&1 &
# --header_channel 1504 \
# CUDA_VISIBLE_DEVICES=5 nohup python -u train.py > txt 2>&1 &



# CUDA_VISIBLE_DEVICES=3 nohup python -u train.py \
# --ngpus_per_node 8 \
# --dataset cifar100 \
# --search_space AddShift \
# --dataset_path /Datadisk/datasets/CIFAR100 \
# --pretrain /Datadisk/shihh/NAS/baseline/CIFAR100_AddShift_q_scratch \
# --nepochs 650 \
# --batch_size 196 \
# --lr 0.02 \
# --load_path /Datadisk/shihh/NAS/hw/CIFAR100_AddShift \
# --load_epoch 240 \
# --gpu 0 > baseline/CIFAR100_AddShift_q_scratch 2>&1 &

# --distillation True \
# --transfer_epoch 400 \


# CUDA_VISIBLE_DEVICES=0 nohup python -u train.py \
# --ngpus_per_node 8 \
# --dataset cifar100 \
# --search_space OnlyConv \
# --dataset_path /Datadisk/datasets/CIFAR100 \
# --pretrain /Datadisk/shihh/NAS/hw/CIFAR100_OnlyConv_distill_5e-3 \
# --nepochs 650 \
# --batch_size 196 \
# --distillation True \
# --lr 0.02 \
# --load_path /Datadisk/shihh/NAS/hw/CIFAR100_OnlyConv \
# --load_epoch 240 \
# --gpu 0 > hw/CIFAR100_OnlyConv_distill_150_5e-3 2>&1 &