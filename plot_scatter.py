
# import matplotlib; matplotlib.use('agg')
import matplotlib
import matplotlib.pyplot as plt
import os, sys
import numpy as np
import pickle
import torch
import math


# pretrain_epoch = [30,50,70,90,100,110]
pretrain_epoch = [30,50,70,90,110,130,150]
acc_cifar10 = [94.947,94.947,94.947,94.947,94.947,94.947]
acc_addshift_cifar10 = [93.23, 93.21, 93.279, 93.299, 94.564, 94.888]
acc_addshiftadd_cifar10 = [92.867, 94.152, 94.211, 94.712, 94.996, 95.065]
acc_addall_cifar10 = [93.465, 93.681, 92.759, 93.75, 92.455]
acc_addall_modify_cifar10 = [91.621, 93.603, 93.495, 94.25, 95.016, 94.927]
acc_cifar100 = [76.903,76.903,76.903,76.903,76.903,76.903]
acc_addshift_cifar100 = [74.10, 74.362, 75.383, 75.893, 76.285, 74.755]
acc_addshiftadd_cifar100 = [68.652, 73.067, 77.051, 75.157, 77.159, 74.814]
# acc_addall_cifar100 = [67.524, 73.057, 69.378, 72.645, 70.948, 73.46]
acc_cifar100_2 = [76.903,76.903,76.903,76.903,76.903,76.903,76.903]
acc_addall_cifar100 = [65.914, 68.515, 71.026, 71.860, 70.673, 73.077, 73.558]

# lw = 8
# font_big = 14
# font_mid = 6.8
# font_small = 10
# font_board = 2
lw = 3
font_big = 14
font_mid = 8
font_small = 7
font_board = 2

# line_type = [['g', 'royalblue'], ['c', 'green'], ['violet', 'pink']]


font1 = {'family' : 'Calibri',
'weight' : 'bold',
'size'   : 15,
}

line_type = [['purple', 'r'], ['c', 'green'], ['royalblue', 'royalblue'],]

"""
acc. vs. ratio (Layerwiser pretrain vs. Random initialization)
"""
# fig, ax = plt.subplots(1,2,figsize=(11, 5))
fig, ax = plt.subplots(figsize=(5, 5))

color1 = 'tab:blue'
color2 = 'tab:green'
# #######################cifar10###############################################
# ax[0].scatter([83.97828657],[94.859], c=line_type[0][1], marker='o', label=r"$\bf{OnlyConv}$", lw=lw)
# ax[0].scatter([114.3149789,93.27244508],[94.447,94.839], c=line_type, marker='+', label=r"$\bf{AddShift}$", lw=2*lw)
# ax[0].scatter([102.5262467,109.4976374,87.54114434,84.38647656],[94.182,94.604,94.721,95.006], c=line_type[2][0], marker='x', label=r"$\bf{AddShiftAdd}$", lw=2*lw)
# ax[0].scatter([85.17887564,119.0204144], [95.016,94.878], c=line_type[1][0], marker='^', label=r"$\bf{AddAll\_modify}$", lw=lw)
# # ax.scatter(pretrain_epoch, acc_cifar10, dashes=[6, 2], color='red', label=r"$\bf{Baseline}$", lw=0.5*lw)
# ax[0].set_ylim([92,96])
# my_x_ticks = np.arange(92,96,1)
# ax[0].set_yticks(my_x_ticks)
# ax[0].legend(fontsize=font_mid,handlelength=50,handleheight=10)
# ax[0].set_xticks(np.arange(75,120,20))
# plt.xticks(fontsize=6)
# ax[0].tick_params(axis='x', labelsize=20)
# ax[0].tick_params(axis='y', labelsize=20)
# ax[0].set_xlabel('FPS', fontweight="bold", fontsize=18)
# ax[0].set_ylabel('Accuracy (%)', fontweight="bold", fontsize=18)
# ax[0].set_title('(a) Accuracy vs FPS on CIFAR-10', fontweight="bold", fontsize=18)

# ax[0].spines['bottom'].set_linewidth(font_board)
# ax[0].spines['bottom'].set_color('black')
# ax[0].spines['left'].set_linewidth(font_board)
# ax[0].spines['left'].set_color('black')
# ax[0].spines['top'].set_linewidth(font_board)
# ax[0].spines['top'].set_color('black')
# ax[0].spines['right'].set_linewidth(font_board)
# ax[0].spines['right'].set_color('black')

# leg = ax[0].legend(fontsize=font_big, loc = 'lower left')
# leg.get_frame().set_edgecolor("black")
# leg.get_frame().set_linewidth(2)
# ax[0].grid(axis='both', color='gray', linestyle='-', linewidth=0.3)

# # cifar 100
# ax[1].scatter([56.41608897],[76.697], c=line_type[0][1], marker='o', label=r"$\bf{OnlyConv}$", lw=lw)
# ax[1].scatter([107.0792215,148.8733267],[75.942,75.805], c=line_type, marker='+', label=r"$\bf{AddShift}$", lw=2*lw)
# ax[1].scatter([82.90004244,94.44511605],[76.737,76.619], c=line_type[2][0], marker='x', label=r"$\bf{AddShiftAdd}$", lw=2*lw)
# # ax[0][1].scatter(pretrain_epoch, acc_addshift_cifar10, c=line_type[3][1], marker='s', label=r"$\bf{AddShift}$", lw=lw)
# ax[1].set_ylim([70,80])
# my_x_ticks = np.arange(70,80 ,2)
# ax[1].set_yticks(my_x_ticks)
# ax[1].legend(fontsize=font_mid)
# ax[1].set_xticks(np.arange(50,130,20))
# plt.xticks(fontsize=6)
# ax[1].tick_params(axis='x', labelsize=20)
# ax[1].tick_params(axis='y', labelsize=20)
# ax[1].set_xlabel('FPS', fontweight="bold", fontsize=18)
# ax[1].set_ylabel('Accuracy (%)', fontweight="bold", fontsize=18)
# ax[1].set_title('(b) Accuracy vs FPS on CIFAR-100', fontweight="bold", fontsize=18)


# ax[1].spines['bottom'].set_linewidth(font_board)
# ax[1].spines['bottom'].set_color('black')
# ax[1].spines['left'].set_linewidth(font_board)
# ax[1].spines['left'].set_color('black')
# ax[1].spines['top'].set_linewidth(font_board)
# ax[1].spines['top'].set_color('black')
# ax[1].spines['right'].set_linewidth(font_board)
# ax[1].spines['right'].set_color('black')

# leg = ax[1].legend(fontsize=font_big, loc = 'lower left')
# leg.get_frame().set_edgecolor("black")
# leg.get_frame().set_linewidth(2)

# ax[1].grid(axis='both', color='gray', linestyle='-', linewidth=0.3)

# ax.plot(pretrain_epoch, acc_addall_cifar100, c=line_type[0][1], marker='o', markersize=2*lw, label=r"$\bf{Baseline}$", lw=lw)
# ax.plot(pretrain_epoch, acc_cifar100_2, dashes=[6, 2], color='red', label=r"$\bf{AddAll}$", lw=lw)
# ax.scatter([110,130],[72.174,75.069], c=line_type[0][1], marker='o', label=r"$\bf{OnlyConv}$", lw=lw)
# ax.set_ylim([60,79])
# my_x_ticks = np.arange(60,79, 2)
# ax.set_yticks(my_x_ticks)
# ax.legend(fontsize=font_mid)
# ax.set_xticks(np.arange(20,150,20))
# plt.xticks(fontsize=6)
# plt.yticks(fontsize=7)
# ax.tick_params(axis='x', labelsize=12)
# ax.tick_params(axis='y', labelsize=12)
# ax.set_xlabel('Pretrain Epoch', fontweight="bold", fontsize=12)
# ax.set_ylabel('Accuracy (%)', fontweight="bold",fontsize=12)
# ax.set_title('(a) AddShift on CIFAR-10', fontweight="bold",fontsize=12)

# ax.scatter([0.587925664], [76.903], c='red', marker='o', label=r"$\bf{OnlyConv}$", lw=2*lw)
# ax.scatter([0.5205183968000001],[76.207], c='royalblue', marker='1', label=r"$\bf{AddAdd}$", lw=4*lw)
# ax.scatter([0.5122265712],[76.285], c='purple', marker='p', label=r"$\bf{AddShift}$", lw=2*lw)
# ax.scatter([0.6182210528],[77.159], c='magenta', marker='s', label=r"$\bf{AddShiftAdd}$", lw=lw)
# ax.scatter([0.6200086736],[77.374], c='green', marker='+', label=r"$\bf{AddAll}$", lw=4*lw)

# ax.scatter([0.508578976], [94.947], c='red', marker='o', label=r"$\bf{OnlyConv}$", lw=2*lw)
# ax.scatter([0.6964143968,0.6979918128],[94.888,94.564], c='royalblue', marker='1', label=r"$\bf{AddShift}$", lw=4*lw)
# ax.scatter([0.45098239360,0.47118467199999997,0.4929833856000001,0.5877204608000001],[95.133,94.741,94.761,95.104], c='purple', marker='p', label=r"$\bf{AddAdd}$", lw=2*lw)
# ax.scatter([0.6041470784000001,0.6525836735999999],[95.065,94.996], c='magenta', marker='s', label=r"$\bf{AddShiftAdd}$", lw=lw)
# ax.scatter([0.5147233568],[95.075], c='green', marker='+', label=r"$\bf{AddAll}$", lw=4*lw)

ax.scatter([0.587925664], [76.903], c='red', marker='o', label=r"$\bf{OnlyConv}$", lw=2*lw)
ax.scatter([0.6200086736],[77.374], c='green', marker='+', label=r"$\bf{AddAll Base}$", lw=4*lw)
ax.scatter([0.44211366360,0.5874852668],[75.285,76.894], c='royalblue', marker='+', label=r"$\bf{AddAll Constraint}$", lw=4*lw)
# ax.scatter([0.5874852668],[76.894], c='green', marker='+', label=r"$\bf{AddAll Base}$", lw=4*lw)

# ax.set_ylim([93.5,96])
# my_x_ticks = np.arange(93.5,96,0.5)
# ax.set_yticks(my_x_ticks)
# ax.legend(fontsize=font_mid)
# ax.set_xticks(np.arange(0.45,0.75, 0.05))
ax.set_ylim([75,78])
my_x_ticks = np.arange(75,78,0.5)
ax.set_yticks(my_x_ticks)
ax.legend(fontsize=font_mid)
ax.set_xticks(np.arange(0.5,0.7, 0.05))
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
ax.tick_params(axis='x', labelsize=12)
ax.tick_params(axis='y', labelsize=12)
ax.set_xlabel('Energy', fontweight="bold", fontsize=12)
ax.set_ylabel('Accuracy (%)', fontweight="bold",fontsize=12)
ax.set_title('(a) CIFAR-100 AddAll', fontweight="bold",fontsize=12)

ax.spines['bottom'].set_linewidth(font_board)
ax.spines['bottom'].set_color('black')
ax.spines['left'].set_linewidth(font_board)
ax.spines['left'].set_color('black')
ax.spines['top'].set_linewidth(font_board)
ax.spines['top'].set_color('black')
ax.spines['right'].set_linewidth(font_board)
ax.spines['right'].set_color('black')

leg = ax.legend(fontsize=font_mid, loc = 'lower left')
leg.get_frame().set_edgecolor("black")
leg.get_frame().set_linewidth(2)
ax.grid(axis='both', color='gray', linestyle='-', linewidth=0.3)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()
plt.savefig('flops.pdf')
plt.close()

