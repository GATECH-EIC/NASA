# CUDA_VISIBLE_DEVICES=6 nohup python -u train.py \
# --ngpus_per_node 8 \
# --dataset cifar10 \
# --search_space AddShift \
# --dataset_path /Datadisk/datasets/CIFAR10 \
# --pretrain /Datadisk/shihh/NAS/baseline/CIFAR10_AddShift_2_1.5e-10_scratch_q_$i \
# --nepochs 650 \
# --batch_size 196 \
# --lr 0.02 \
# --load_path /Datadisk/shihh/NAS/baseline/CIFAR10_AddShift_2_1.5e-10 \
# --load_epoch 130 \
# --gpu 0 > baseline/CIFAR10_AddShift_2_1.5e-10_scratch_q_$i 2>&1 &
# --header_channel 1504 \
# CUDA_VISIBLE_DEVICES=5 nohup python -u train.py > txt 2>&1 &

# cuda=(3 2)
# for i in 0 1 2
# do
#     CUDA_VISIBLE_DEVICES=$i nohup python -u train.py \
#     --ngpus_per_node 1 \
#     --dataset cifar10 \
#     --search_space AddAll \
#     --dataset_path /media/HD1/cifar/CIFAR10 \
#     --pretrain /media/HD1/shh/NAS/ckpt/CIFAR10_AddAll_2_1_scratch_q_augment_$i \
#     --nepochs 650 \
#     --batch_size 196 \
#     --lr_schedule multistep \
#     --lr 0.1 \
#     --weight_decay 2e-4 \
#     --resume_path /media/HD1/shh/NAS/ckpt/CIFAR10_AddAll_2_1_scratch_q_augment_$i \
#     --load_path /ckpt/CIFAR10_AddAll_C_$i \
#     --load_epoch 210 \
#     --gpu 0 > AddAll/CIFAR10_AddAll_C_$i 2>&1 &
# done

# --distillation True \
# --transfer_epoch 400 \


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


for i in 0 1 2
do
    CUDA_VISIBLE_DEVICES=$i nohup python -u train.py \
    --ngpus_per_node 1 \
    --dataset cifar10 \
    --search_space AddAll \
    --dataset_path /media/HD1/cifar/CIFAR10 \
    --nepochs 650 \
    --batch_size 196 \
    --lr_schedule multistep \
    --lr 0.1 \
    --weight_decay 2e-4 \
    --load_path /ckpt/CIFAR10_AddAll_C_$i \
    --load_epoch 210 \
    --gpu 0 > AddAll/CIFAR10_AddAll_C_$i 2>&1 &
done