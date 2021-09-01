
# import matplotlib; matplotlib.use('agg')
import matplotlib
import matplotlib.pyplot as plt
import os, sys
import numpy as np
import pickle
import torch
import math


pretrain_epoch = [30,50,70,90,100,110]
pretrain_epoch_2 = [30,50,70,90,100,110,130,150]
acc_cifar10 = [94.947,94.947,94.947,94.947,94.947,94.947]
acc_cifar10_2 = [94.947,94.947,94.947,94.947,94.947,94.947,94.947,94.947]
acc_addshift_cifar10 = [93.23, 93.21, 93.279, 93.299, 94.564, 94.888]
acc_addshift_cifar10_4 = [93.819, 93.377, 94.505, 95.222, 95.094, 94.653]
acc_addshiftadd_cifar10 = [92.867, 94.152, 94.211, 94.712, 94.996, 95.065]
acc_addall_cifar10 = [93.465, 93.681, 92.759, 93.75, 92.455]
acc_addall_modify_cifar10 = [91.621, 93.603, 93.495, 94.25, 95.016, 94.927]
acc_addall_noshiftadd_cifar10 = [91.591, 93.77, 92.955, 93.936, 94.525, 95.241, 95.075, 94.751]
acc_addall_noshiftadd_4_cifar10 = [94.299, 92.465, 94.299, 92.857, 93.936, 94.162, 94.152, 94.78]
acc_addadd_cifar10_4 = [93.23, 93.77, 94.172, 94.192, 94.26, 94.761, 94.741, 95.104]
acc_addadd_cifar10 = [93.848, 94.113, 94.506, 94.113, 94.525, 95.133, 95.143, 95.241]

acc_cifar100 = [76.903,76.903,76.903,76.903,76.903,76.903]
acc_cifar100_2 = [76.903,76.903,76.903,76.903,76.903,76.903,76.903,76.903]
acc_addshift_cifar100 = [74.107, 74.362, 75.383, 75.893, 76.285, 74.755, 72.734, 72.106]
acc_addshift_cifar100_4 = [72.292, 72.272, 74.225, 75.157, 75.245, 76.354, 77.168, 76.236]
acc_addshiftadd_cifar100 = [68.652, 73.067, 77.051, 75.157, 77.159, 74.814]
acc_addall_cifar100 = [67.524, 73.057, 69.378, 72.645, 70.948, 73.46]
acc_addall_noshiftadd_cifar100 = [63.746, 62.726, 68.181, 68.534, 73.95, 72.047, 70.909, 70.447]
acc_addall_noshiftadd_4_cifar100 = [68.093, 68.308, 68.367, 71.085, 70.634, 73.675, 70.546, 68.858]
acc_addadd_cifar100 = [67.347, 72.017, 73.518, 74.961, 74.725, 75.54, 75.442, 75.814]
acc_addadd_cifar100_4 = [70.065, 71.311, 74.5, 76.168, 76.207, 76.187, 74.666, 76.246]

lw = 3
font_big = 14
font_mid = 10
font_small = 7
font_board = 2

line_type = [['g', 'royalblue'], ['c', 'green'], ['violet', 'pink']]

"""
acc. vs. ratio (Layerwiser pretrain vs. Random initialization)
"""
fig, ax = plt.subplots(2,3,figsize=(12, 5))

color1 = 'tab:blue'
color2 = 'tab:green'
#######################cifar10###############################################
ax[0][0].plot(pretrain_epoch, acc_cifar10, dashes=[6, 2], color='red', label=r"$\bf{Baseline}$", lw=0.5*lw)
ax[0][0].plot(pretrain_epoch, acc_addshift_cifar10, c=line_type[0][1], marker='o', markersize=2*lw, label=r"$\bf{AddShift3}$", lw=lw)
ax[0][0].plot(pretrain_epoch, acc_addshift_cifar10_4, c=line_type[1][1], marker='o', markersize=2*lw, label=r"$\bf{AddShift4}$", lw=lw)
ax[0][0].set_ylim([90,96])
my_x_ticks = np.arange(90,96 ,1)
ax[0][0].set_yticks(my_x_ticks)
ax[0][0].legend(fontsize=font_mid)
ax[0][0].set_xticks(np.arange(20,130,20))
plt.xticks(fontsize=6)
plt.yticks(fontsize=7)
ax[0][0].tick_params(axis='x', labelsize=12)
ax[0][0].tick_params(axis='y', labelsize=12)
ax[0][0].set_xlabel('Pretrain Epoch', fontweight="bold", fontsize=12)
ax[0][0].set_ylabel('Accuracy (%)', fontweight="bold",fontsize=12)
ax[0][0].set_title('(a) AddShift on CIFAR-10', fontweight="bold",fontsize=12)

ax[0][0].spines['bottom'].set_linewidth(font_board)
ax[0][0].spines['bottom'].set_color('black')
ax[0][0].spines['left'].set_linewidth(font_board)
ax[0][0].spines['left'].set_color('black')
ax[0][0].spines['top'].set_linewidth(font_board)
ax[0][0].spines['top'].set_color('black')
ax[0][0].spines['right'].set_linewidth(font_board)
ax[0][0].spines['right'].set_color('black')

leg = ax[0][0].legend(fontsize=font_mid, loc = 'lower left')
leg.get_frame().set_edgecolor("black")
leg.get_frame().set_linewidth(2)
ax[0][0].grid(axis='both', color='gray', linestyle='-', linewidth=0.3)


#########################
ax[0][1].plot(pretrain_epoch_2, acc_cifar10_2, dashes=[6, 2], color='red', label=r"$\bf{Baseline}$", lw=0.5*lw)
ax[0][1].plot(pretrain_epoch_2, acc_addadd_cifar10, c=line_type[0][1], marker='o', markersize=2*lw, label=r"$\bf{AddAdd3}$", lw=lw)
ax[0][1].plot(pretrain_epoch_2, acc_addadd_cifar10_4, c=line_type[1][1], marker='o', markersize=2*lw, label=r"$\bf{AddAdd4}$", lw=lw)
ax[0][1].set_ylim([90,96])
my_x_ticks = np.arange(90,96,1)
ax[0][1].set_yticks(my_x_ticks)
ax[0][1].legend(fontsize=font_mid)
ax[0][1].set_xticks(np.arange(20,170,20))
plt.xticks(fontsize=3)
ax[0][1].tick_params(axis='x', labelsize=12)
ax[0][1].tick_params(axis='y', labelsize=12)
ax[0][1].set_xlabel('Pretrain Epoch', fontweight="bold", fontsize=12)
ax[0][1].set_ylabel('Accuracy (%)', fontweight="bold",fontsize=12)
ax[0][1].set_title('(b) AddAdd on CIFAR-10', fontweight="bold",fontsize=12)

ax[0][1].spines['bottom'].set_linewidth(font_board)
ax[0][1].spines['bottom'].set_color('black')
ax[0][1].spines['left'].set_linewidth(font_board)
ax[0][1].spines['left'].set_color('black')
ax[0][1].spines['top'].set_linewidth(font_board)
ax[0][1].spines['top'].set_color('black')
ax[0][1].spines['right'].set_linewidth(font_board)
ax[0][1].spines['right'].set_color('black')

leg = ax[0][1].legend(fontsize=font_mid, loc = 'lower left')
leg.get_frame().set_edgecolor("black")
leg.get_frame().set_linewidth(2)

ax[0][1].grid(axis='both', color='gray', linestyle='-', linewidth=0.3)
# cifar 100
# ax[0][1].plot(pretrain_epoch, acc_addshiftadd_cifar10, c=line_type[0][1], marker='o', markersize=2*lw, label=r"$\bf{AddShiftAdd}$", lw=lw)
# ax[0][1].plot(pretrain_epoch, acc_cifar10, dashes=[6, 2], color='red', label=r"$\bf{Baseline}$", lw=0.5*lw)
# ax[0][1].set_ylim([90,96])
# my_x_ticks = np.arange(90,96 ,2)
# ax[0][1].set_yticks(my_x_ticks)
# ax[0][1].legend(fontsize=font_mid)
# ax[0][1].set_xticks(np.arange(20,130,20))
# plt.xticks(fontsize=3)
# ax[0][1].tick_params(axis='x', labelsize=12)
# ax[0][1].tick_params(axis='y', labelsize=12)
# ax[0][1].set_xlabel('Pretrain Epoch', fontweight="bold", fontsize=12)
# ax[0][1].set_ylabel('Accuracy (%)', fontweight="bold",fontsize=12)
# ax[0][1].set_title('(b) AddShiftAdd on CIFAR-10', fontweight="bold",fontsize=12)

# ax[0][1].spines['bottom'].set_linewidth(font_board)
# ax[0][1].spines['bottom'].set_color('black')
# ax[0][1].spines['left'].set_linewidth(font_board)
# ax[0][1].spines['left'].set_color('black')
# ax[0][1].spines['top'].set_linewidth(font_board)
# ax[0][1].spines['top'].set_color('black')
# ax[0][1].spines['right'].set_linewidth(font_board)
# ax[0][1].spines['right'].set_color('black')

# leg = ax[0][1].legend(fontsize=font_mid, loc = 'lower left')
# leg.get_frame().set_edgecolor("black")
# leg.get_frame().set_linewidth(2)

# ax[0][1].grid(axis='both', color='gray', linestyle='-', linewidth=0.3)

#######################################################################
# ax[0][2].plot([30,50,70,100,110], acc_addall_cifar10, c=line_type[2][0], marker='o', markersize=2*lw, label=r"$\bf{AddAll}$", lw=lw)
ax[0][2].plot(pretrain_epoch, acc_addall_modify_cifar10, c=line_type[1][0], marker='*', markersize=2*lw, label=r"$\bf{AddAll\_modify}$", lw=lw)
ax[0][2].plot(pretrain_epoch_2, acc_addall_noshiftadd_cifar10, c=line_type[0][0], marker='o', markersize=2*lw, label=r"$\bf{AddAll3}$", lw=lw)
ax[0][2].plot(pretrain_epoch_2, acc_addall_noshiftadd_4_cifar10, c=line_type[0][1], marker='^', markersize=2*lw, label=r"$\bf{AddAll4}$", lw=lw)
ax[0][2].plot(pretrain_epoch_2, acc_cifar10_2, dashes=[6, 2], color='red', label=r"$\bf{Baseline}$", lw=0.5*lw)
ax[0][2].set_ylim([90,96])
my_x_ticks = np.arange(90,96 ,1)
ax[0][2].set_yticks(my_x_ticks)
ax[0][2].legend(fontsize=font_mid)
ax[0][2].set_xticks(np.arange(20,170,20))
plt.xticks(fontsize=3)
ax[0][2].tick_params(axis='x', labelsize=12)
ax[0][2].tick_params(axis='y', labelsize=12)
ax[0][2].set_xlabel('Pretrain Epoch', fontweight="bold", fontsize=12)
ax[0][2].set_ylabel('Accuracy (%)', fontweight="bold",fontsize=12)
ax[0][2].set_title('(c) AddAll on CIFAR-10', fontweight="bold",fontsize=12)

ax[0][2].spines['bottom'].set_linewidth(font_board)
ax[0][2].spines['bottom'].set_color('black')
ax[0][2].spines['left'].set_linewidth(font_board)
ax[0][2].spines['left'].set_color('black')
ax[0][2].spines['top'].set_linewidth(font_board)
ax[0][2].spines['top'].set_color('black')
ax[0][2].spines['right'].set_linewidth(font_board)
ax[0][2].spines['right'].set_color('black')

leg = ax[0][2].legend(fontsize=font_small, loc = 'lower right')
leg.get_frame().set_edgecolor("black")
leg.get_frame().set_linewidth(2)
ax[0][2].grid(axis='both', color='gray', linestyle='-', linewidth=0.3)


#############################cifar100########################################
ax[1][0].plot(pretrain_epoch_2, acc_cifar100_2, dashes=[6, 2], color='red', label=r"$\bf{Baseline}$", lw=0.5*lw)
ax[1][0].plot(pretrain_epoch_2, acc_addshift_cifar100, c=line_type[0][1], marker='o', markersize=2*lw, label=r"$\bf{AddShift3}$", lw=lw)
ax[1][0].plot(pretrain_epoch_2, acc_addshift_cifar100_4, c=line_type[1][1], marker='o', markersize=2*lw, label=r"$\bf{AddShift4}$", lw=lw)
ax[1][0].set_ylim([63, 80])
my_x_ticks = np.arange(63, 80,2)
ax[1][0].set_yticks(my_x_ticks)
ax[1][0].legend(fontsize=font_mid)
ax[1][0].set_xticks(np.arange(20,170,20))
plt.xticks(fontsize=3)
ax[1][0].tick_params(axis='x', labelsize=12)
ax[1][0].tick_params(axis='y', labelsize=12)
ax[1][0].set_xlabel('Pretrain Epoch', fontweight="bold", fontsize=12)
ax[1][0].set_ylabel('Accuracy (%)', fontweight="bold",fontsize=12)
ax[1][0].set_title('(d) AddShift on CIFAR-100', fontweight="bold",fontsize=12)

ax[1][0].spines['bottom'].set_linewidth(font_board)
ax[1][0].spines['bottom'].set_color('black')
ax[1][0].spines['left'].set_linewidth(font_board)
ax[1][0].spines['left'].set_color('black')
ax[1][0].spines['top'].set_linewidth(font_board)
ax[1][0].spines['top'].set_color('black')
ax[1][0].spines['right'].set_linewidth(font_board)
ax[1][0].spines['right'].set_color('black')

leg = ax[1][0].legend(fontsize=font_mid, loc = 'lower left')
leg.get_frame().set_edgecolor("black")
leg.get_frame().set_linewidth(2)
ax[1][0].grid(axis='both', color='gray', linestyle='-', linewidth=0.3)

# cifar 100
# ax[1][1].plot(pretrain_epoch, acc_addshiftadd_cifar100, c=line_type[0][1], marker='o', markersize=2*lw, label=r"$\bf{AddShiftAdd}$", lw=lw)
# ax[1][1].plot(pretrain_epoch, acc_cifar100, dashes=[6, 2], color='red', label=r"$\bf{Baseline}$", lw=0.5*lw)
# ax[1][1].set_ylim([60, 79])
# my_x_ticks = np.arange(60 ,79 ,2)
# ax[1][1].set_yticks(my_x_ticks)
# ax[1][1].legend(fontsize=font_mid)
# ax[1][1].set_xticks(np.arange(20,130,20))
# plt.xticks(fontsize=3)
# ax[1][1].tick_params(axis='x', labelsize=12)
# ax[1][1].tick_params(axis='y', labelsize=12)
# ax[1][1].set_xlabel('Pretrain Epoch', fontweight="bold", fontsize=12)
# ax[1][1].set_ylabel('Accuracy (%)', fontweight="bold",fontsize=12)
# ax[1][1].set_title('(e) AddShift on CIFAR-100', fontweight="bold",fontsize=12)

# ax[1][1].spines['bottom'].set_linewidth(font_board)
# ax[1][1].spines['bottom'].set_color('black')
# ax[1][1].spines['left'].set_linewidth(font_board)
# ax[1][1].spines['left'].set_color('black')
# ax[1][1].spines['top'].set_linewidth(font_board)
# ax[1][1].spines['top'].set_color('black')
# ax[1][1].spines['right'].set_linewidth(font_board)
# ax[1][1].spines['right'].set_color('black')

# leg = ax[1][1].legend(fontsize=font_mid, loc = 'lower left')
# leg.get_frame().set_edgecolor("black")
# leg.get_frame().set_linewidth(2)

# ax[1][1].grid(axis='both', color='gray', linestyle='-', linewidth=0.3)

#######################################################################
ax[1][1].plot(pretrain_epoch_2, acc_cifar100_2, dashes=[6, 2], color='red', label=r"$\bf{Baseline}$", lw=0.5*lw)
ax[1][1].plot(pretrain_epoch_2, acc_addadd_cifar100, c=line_type[0][1], marker='o', markersize=2*lw, label=r"$\bf{AddAdd3}$", lw=lw)
ax[1][1].plot(pretrain_epoch_2, acc_addadd_cifar100_4, c=line_type[1][1], marker='o', markersize=2*lw, label=r"$\bf{AddAdd4}$", lw=lw)
ax[1][1].set_ylim([63, 79])
my_x_ticks = np.arange(63 ,79 ,2)
ax[1][1].set_yticks(my_x_ticks)
ax[1][1].legend(fontsize=font_mid)
ax[1][1].set_xticks(np.arange(20,160,20))
plt.xticks(fontsize=3)
ax[1][1].tick_params(axis='x', labelsize=12)
ax[1][1].tick_params(axis='y', labelsize=12)
ax[1][1].set_xlabel('Pretrain Epoch', fontweight="bold", fontsize=12)
ax[1][1].set_ylabel('Accuracy (%)', fontweight="bold",fontsize=12)
ax[1][1].set_title('(e) AddAdd on CIFAR-100', fontweight="bold",fontsize=12)

ax[1][1].spines['bottom'].set_linewidth(font_board)
ax[1][1].spines['bottom'].set_color('black')
ax[1][1].spines['left'].set_linewidth(font_board)
ax[1][1].spines['left'].set_color('black')
ax[1][1].spines['top'].set_linewidth(font_board)
ax[1][1].spines['top'].set_color('black')
ax[1][1].spines['right'].set_linewidth(font_board)
ax[1][1].spines['right'].set_color('black')

leg = ax[1][1].legend(fontsize=font_mid, loc = 'lower right')
leg.get_frame().set_edgecolor("black")
leg.get_frame().set_linewidth(2)

ax[1][1].grid(axis='both', color='gray', linestyle='-', linewidth=0.3)

#####################################################
ax[1][2].plot(pretrain_epoch, acc_addall_cifar100, c=line_type[2][0], marker='*', markersize=2*lw, label=r"$\bf{AddAll}$", lw=lw)
ax[1][2].plot(pretrain_epoch_2, acc_addall_noshiftadd_cifar100, c=line_type[0][0], marker='o', markersize=2*lw, label=r"$\bf{AddAll3}$", lw=lw)
ax[1][2].plot(pretrain_epoch_2, acc_addall_noshiftadd_4_cifar100, c=line_type[0][1], marker='o', markersize=2*lw, label=r"$\bf{AddAll4}$", lw=lw)
ax[1][2].plot(pretrain_epoch_2, acc_cifar100_2, dashes=[6, 2], color='red', label=r"$\bf{Baseline}$", lw=0.5*lw)
ax[1][2].set_ylim([53, 79])
my_x_ticks = np.arange(53, 79 ,3)
ax[1][2].set_yticks(my_x_ticks)
ax[1][2].legend(fontsize=font_mid)
ax[1][2].set_xticks(np.arange(20,170,20))
plt.xticks(fontsize=3)
ax[1][2].tick_params(axis='x', labelsize=12)
ax[1][2].tick_params(axis='y', labelsize=12)
ax[1][2].set_xlabel('Pretrain Epoch', fontweight="bold", fontsize=12)
ax[1][2].set_ylabel('Accuracy (%)', fontweight="bold",fontsize=12)
ax[1][2].set_title('(f) AddAll on CIFAR-100', fontweight="bold",fontsize=12)

ax[1][2].spines['bottom'].set_linewidth(font_board)
ax[1][2].spines['bottom'].set_color('black')
ax[1][2].spines['left'].set_linewidth(font_board)
ax[1][2].spines['left'].set_color('black')
ax[1][2].spines['top'].set_linewidth(font_board)
ax[1][2].spines['top'].set_color('black')
ax[1][2].spines['right'].set_linewidth(font_board)
ax[1][2].spines['right'].set_color('black')

leg = ax[1][2].legend(fontsize=font_mid, loc = 'lower right')
leg.get_frame().set_edgecolor("black")
leg.get_frame().set_linewidth(2)
ax[1][2].grid(axis='both', color='gray', linestyle='-', linewidth=0.3)

plt.tight_layout()
plt.show()
plt.savefig('nas_result.pdf')
plt.close()


"""
# acc vs. flops (EB Train vs. Layerwise pretrain)
# """
# fig, ax = plt.subplots(1,2,figsize=(6, 2.5))

# color1 = 'tab:blue'
# color2 = 'tab:green'

# x1 = []
# x2 = []
# for i,j in zip(flops_10_layer,flops_10_eb):
# 	# x1.append(i/1E5)
# 	# x2.append(j/1E5)
#     x1.append(i)
#     x2.append(j)

# ax.plot(x1, acc_10_layer, c=line_type[0][1], marker='o', markersize=2*lw, label=r"$\bf{Layerwise\ Pretrain}$", lw=lw)
# ax.plot(x2, acc_10_eb, c=line_type[0][0], marker='^', markersize=2*lw, label=r"$\bf{Lottery\ Ticket}$", lw=lw)

# ax.set_ylim([88, 96])


# my_x_ticks = np.arange(88, 96, 2)
# ax.set_yticks(my_x_ticks)


# ax.legend(fontsize=font_mid)
# # ax.set_yscale('log')
# mii = min(math.floor(min(x1)),math.floor(min(x2)))
# maa = max(math.ceil(max(x1)),math.ceil(max(x2)))
# ax.set_xticks(np.arange(mii,maa,int((maa-mii)/5)))
# plt.xticks(fontsize=3)
# ax.tick_params(axis='x', labelsize=8)
# # ax.set_xlabel('FLOPs (1e5)', fontweight="bold")
# ax.set_xlabel('FLOPs', fontweight="bold")
# ax.set_ylabel('Testing Accuracy (%)', fontweight="bold")

# ax.spines['bottom'].set_linewidth(font_board)
# ax.spines['bottom'].set_color('black')
# ax.spines['left'].set_linewidth(font_board)
# ax.spines['left'].set_color('black')
# ax.spines['top'].set_linewidth(font_board)
# ax.spines['top'].set_color('black')
# ax.spines['right'].set_linewidth(font_board)
# ax.spines['right'].set_color('black')

# ax.set_title('VGG-16 on CIFAR-10', fontweight="bold")
# ax.grid(axis='both', color='gray', linestyle='-', linewidth=0.3)

# leg = ax.legend(fontsize=font_mid, loc = 'lower right')
# leg.get_frame().set_edgecolor("black")
# leg.get_frame().set_linewidth(2)

# # CIFAR-100

# color1 = 'tab:blue'
# color2 = 'tab:green'

# x1 = []
# x2 = []
# for i,j in zip(flops_100_layer,flops_100_eb):
# 	# x1.append(i/1E5)
# 	# x2.append(j/1E5)
#     x1.append(i)
#     x2.append(j)

# ax[1].plot(x1, acc_100_layer, c=line_type[0][1], marker='o', markersize=2*lw, label=r"$\bf{Layerwise\ Pretrain}$", lw=lw)
# ax[1].plot(x2, acc_100_eb, c=line_type[0][0], marker='^', markersize=2*lw, label=r"$\bf{Lottery\ Ticket}$", lw=lw)

# ax[1].set_ylim([64, 74])


# my_x_ticks = np.arange(64, 74, 2)
# ax[1].set_yticks(my_x_ticks)


# ax[1].legend(fontsize=font_mid)
# # ax.set_yscale('log')
# mii = min(math.floor(min(x1)),math.floor(min(x2)))
# maa = max(math.ceil(max(x1)),math.ceil(max(x2)))
# ax[1].set_xticks(np.arange(mii,maa,int((maa-mii)/5)))
# plt.xticks(fontsize=3)
# ax[1].tick_params(axis='x', labelsize=8)
# # ax[1].set_xlabel('FLOPs (1e5)', fontweight="bold")
# ax[1].set_xlabel('FLOPs', fontweight="bold")
# ax[1].set_ylabel('Testing Accuracy (%)', fontweight="bold")

# ax[1].spines['bottom'].set_linewidth(font_board)
# ax[1].spines['bottom'].set_color('black')
# ax[1].spines['left'].set_linewidth(font_board)
# ax[1].spines['left'].set_color('black')
# ax[1].spines['top'].set_linewidth(font_board)
# ax[1].spines['top'].set_color('black')
# ax[1].spines['right'].set_linewidth(font_board)
# ax[1].spines['right'].set_color('black')

# ax[1].set_title('VGG-16 on CIFAR-100', fontweight="bold")

# leg = ax[1].legend(fontsize=font_mid, loc = 'lower right')
# leg.get_frame().set_edgecolor("black")
# leg.get_frame().set_linewidth(2)

# ax[1].grid(axis='both', color='gray', linestyle='-', linewidth=0.3)
# plt.tight_layout()
# plt.show()
# plt.savefig('EB-Layerwise.pdf')
# plt.close()