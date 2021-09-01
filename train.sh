CUDA_VISIBLE_DEVICES=1 nohup python -u train.py \
--ngpus_per_node 8 \
--dataset cifar100 \
--search_space AddShift \
--dataset_path /Datadisk/datasets/CIFAR100 \
--pretrain /Datadisk/shihh/NAS/baseline/CIFAR100_AddShift_q_scratch \
--nepochs 650 \
--batch_size 196 \
--lr 0.02 \
--load_path /Datadisk/shihh/NAS/hw/CIFAR100_AddShift \
--load_epoch 240 \
--gpu 0 > baseline/CIFAR100_AddShift_q_scratch_150 2>&1 &
# --header_channel 1504 \
# CUDA_VISIBLE_DEVICES=5 nohup python -u train.py > txt 2>&1 &

# CUDA_VISIBLE_DEVICES=0 nohup python -u train.py \
# --ngpus_per_node 8 \
# --dataset cifar10 \
# --search_space AddShift \
# --dataset_path /Datadisk/datasets/CIFAR10 \
# --pretrain /Datadisk/shihh/NAS/baseline/CIFAR10_AddShift_scratch \
# --nepochs 650 \
# --batch_size 196 \
# --lr 0.02 \
# --load_path /Datadisk/shihh/NAS/hw/CIFAR10_AddShift2 \
# --load_epoch 220 \
# --gpu 0 > hw/CIFAR10_AddShift2_scratch_130 2>&1 &

# CUDA_VISIBLE_DEVICES=0 nohup python -u train.py \
# --ngpus_per_node 8 \
# --dataset cifar100 \
# --search_space AddShift \
# --dataset_path /Datadisk/datasets/CIFAR100 \
# --pretrain /Datadisk/shihh/NAS/hw/CIFAR100_AddShift_distill_5e-3 \
# --nepochs 650 \
# --batch_size 196 \
# --distillation True \
# --lr 0.02 \
# --load_path /Datadisk/shihh/NAS/hw/CIFAR100_AddShift \
# --load_epoch 240 \
# --gpu 0 > hw/CIFAR100_AddShift_distill_150_5e-3 2>&1 &