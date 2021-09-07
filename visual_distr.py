# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker
# import deepshift.ste as ste

# def shift_process_ps(shift,sign):
#     shift_clip = ste.clamp(shift, *((-1 * (2**(6 - 1) - 2), 0)))
#     shift_rounded = ste.round(shift_clip, rounding='deterministic')
#     sign_rounded_signed = ste.sign(ste.round(sign, 'deterministic'))
#     # sign_rounded_signed = ste.sign(sign)
#     weight_ps = ste.unsym_grad_mul(2**shift_rounded, sign_rounded_signed)
#     return weight_ps
# # weights.append(weight_ps)

# def shift_process_q(weight):
#     weight = ste.clampabs(weight.data, 2**(-1 * (2**(6 - 1) - 1)), 2**0)
#     weight_q = ste.round_power_of_2(weight, 'deterministic')
#     return weight_q

# font1 = {'family' : 'Calibri',
# 'weight' : 'bold',
# 'size'   : 45,
# }
# font_board = 8

# fig, ax = plt.subplots(1,3,figsize=(35, 8))
# pretrained_model = torch.load('/Datadisk/shihh/NAS/baseline/CIFAR100_AddShift_q_scratch/weights_best_240.pth')
# partial = pretrained_model['state_dict']
# weights = []
# shift= []
# sign = []
# # weights.append(partial['module.cells.0._op.conv1.weight'].cpu().detach().numpy())
# # weights.append(partial['module.cells.0._op.conv2.weight'].cpu().detach().numpy())
# # weights.append(partial['module.cells.0._op.conv3.weight'].cpu().detach().numpy())
# # weights.append(partial['module.cells.4._op.conv1.weight'].cpu().detach().numpy())
# # weights.append(partial['module.cells.4._op.conv2.weight'].cpu().detach().numpy())
# # weights.append(partial['module.cells.4._op.conv3.weight'].cpu().detach().numpy())
# # weights.append(partial['module.cells.13._op.conv1.weight'].cpu().detach().numpy())
# # weights.append(partial['module.cells.13._op.conv2.weight'].cpu().detach().numpy())
# # weights.append(partial['module.cells.13._op.conv3.weight'].cpu().detach().numpy())
# # weights.append(shift_process_q(partial['module.cells.14._op.conv1.weight']).cpu().detach().numpy())
# # weights.append(shift_process_q(partial['module.cells.14._op.conv2.weight']).cpu().detach().numpy())
# # weights.append(shift_process_q(partial['module.cells.14._op.conv3.weight']).cpu().detach().numpy())
# # weights.append(shift_process_q(partial['module.cells.15._op.conv1.weight']).cpu().detach().numpy())
# # weights.append(shift_process_q(partial['module.cells.15._op.conv2.weight']).cpu().detach().numpy())
# # weights.append(shift_process_q(partial['module.cells.15._op.conv3.weight']).cpu().detach().numpy())

# # weights.append(shift_process_ps(partial['module.cells.15._op.conv1.shift'],partial['module.cells.15._op.conv1.sign']).cpu().detach().numpy())
# # print(weights)
# # weights.append(shift_process_ps(partial['module.cells.15._op.conv2.shift'],partial['module.cells.15._op.conv2.sign']).cpu().detach().numpy())
# # weights.append(shift_process_ps(partial['module.cells.15._op.conv3.shift'],partial['module.cells.15._op.conv3.sign']).cpu().detach().numpy())
# # weights.append(shift_process_q(partial['module.cells.16._op.conv1.weight']).cpu().detach().numpy())
# # weights.append(shift_process_q(partial['module.cells.16._op.conv2.weight']).cpu().detach().numpy())
# # weights.append(shift_process_q(partial['module.cells.16._op.conv3.weight']).cpu().detach().numpy())
# weights.append(partial['module.cells.17._op.conv1.weight'].cpu().detach().numpy())
# weights.append(partial['module.cells.17._op.conv2.weight'].cpu().detach().numpy())
# weights.append(partial['module.cells.17._op.conv3.weight'].cpu().detach().numpy())
# # weights.append(shift_process_q(partial['module.cells.22._op.conv1.weight']).cpu().detach().numpy())
# # weights.append(shift_process_q(partial['module.cells.22._op.conv2.weight']).cpu().detach().numpy())
# # weights.append(shift_process_q(partial['module.cells.22._op.conv3.weight']).cpu().detach().numpy())
# # weights.append(shift_process_q(partial['module.cells.20._op.conv1.weight']).cpu().detach().numpy())
# # weights.append(shift_process_q(partial['module.cells.20._op.conv2.weight']).cpu().detach().numpy())
# # weights.append(shift_process_q(partial['module.cells.20._op.conv3.weight']).cpu().detach().numpy())

# all_weights = np.zeros([0, ])
# w_flatten = []
# plot_weights = []
# text = ['PW1', 'DW', 'PW2']
# x_range = [[-0.05,0.05],[-0.05,0.05],[-0.05,0.05],[-0.01,0.01],[-0.01,0.01],[-0.01,0.01],[-0.05,0.05],[-0.01,0.01],[-0.01,0.01]]
# for j in range(1):
#     for i in range (3):
#         w_flatten.append(np.reshape(weights[3*j+i], [-1]))
#         plot_weights.append(np.concatenate([all_weights, w_flatten[3*j+i]], axis=0))
#         ax[i].hist(plot_weights[3*j+i], bins=100, color="royalblue",range=[-0.05,0.05])
#         # ax[j][i].xaxis.set_major_locator(ticker.MultipleLocator(1))
#         ax[i].tick_params(axis='x', labelsize=45)
#         ax[i].tick_params(axis='y', labelsize=45)
#         ax[i].set_xlabel('Weight', fontweight="bold", fontsize=45)
#         ax[i].set_ylabel('Count', fontweight="bold",fontsize=45)
#         ax[i].set_title(text[i],fontdict=font1)  
#         # ax[i].set_title('(a) CIFAR-10', fontweight="bold",fontsize=12)
#         ax[i].ticklabel_format(style='sci',scilimits=(0,0),axis='y')
#         ax[i].spines['bottom'].set_linewidth(font_board)
#         ax[i].spines['bottom'].set_color('black')
#         ax[i].spines['left'].set_linewidth(font_board)
#         ax[i].spines['left'].set_color('black')
#         ax[i].spines['top'].set_linewidth(font_board)
#         ax[i].spines['top'].set_color('black')
#         ax[i].spines['right'].set_linewidth(font_board)
#         ax[i].spines['right'].set_color('black')
# plt.tight_layout()
# plt.savefig("visualize/shift_distribution_compare.png")
# plt.clf()

###############################################################
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import deepshift.ste as ste

def shift_process(shift,sign):
    shift_clip = ste.clamp(shift, *((-1 * (2**(6 - 1) - 2), 0)))
    shift_rounded = ste.round(shift_clip, rounding='deterministic')
    sign_rounded_signed = ste.sign(ste.round(sign, 'deterministic'))
    weight_ps = ste.unsym_grad_mul(2**shift_rounded, sign_rounded_signed)
    return weight_ps


fig, ax = plt.subplots(9,3,figsize=(18, 25))
pretrained_model = torch.load('/Datadisk/shihh/NAS/baseline/CIFAR100_AddAdd_scratch/weights_best_240.pth')
partial = pretrained_model['state_dict']
weights = []

weights.append(partial['module.cells.0._op.conv1.weight'].cpu().detach().numpy())
weights.append(partial['module.cells.0._op.conv2.weight'].cpu().detach().numpy())
weights.append(partial['module.cells.0._op.conv3.weight'].cpu().detach().numpy())
weights.append(partial['module.cells.13._op.conv1.weight'].cpu().detach().numpy())
weights.append(partial['module.cells.13._op.conv2.weight'].cpu().detach().numpy())
weights.append(partial['module.cells.13._op.conv3.weight'].cpu().detach().numpy())
weights.append(partial['module.cells.14._op.conv1.adder'].cpu().detach().numpy())
weights.append(partial['module.cells.14._op.conv2.adder'].cpu().detach().numpy())
weights.append(partial['module.cells.14._op.conv3.adder'].cpu().detach().numpy())
weights.append(partial['module.cells.15._op.conv1.adder'].cpu().detach().numpy())
weights.append(partial['module.cells.15._op.conv2.adder'].cpu().detach().numpy())
weights.append(partial['module.cells.15._op.conv3.adder'].cpu().detach().numpy())
weights.append(partial['module.cells.16._op.conv1.weight'].cpu().detach().numpy())
weights.append(partial['module.cells.16._op.conv2.weight'].cpu().detach().numpy())
weights.append(partial['module.cells.16._op.conv3.weight'].cpu().detach().numpy())
weights.append(partial['module.cells.17._op.conv1.weight'].cpu().detach().numpy())
weights.append(partial['module.cells.17._op.conv2.weight'].cpu().detach().numpy())
weights.append(partial['module.cells.17._op.conv3.weight'].cpu().detach().numpy())
weights.append(partial['module.cells.18._op.conv1.adder'].cpu().detach().numpy())
weights.append(partial['module.cells.18._op.conv2.adder'].cpu().detach().numpy())
weights.append(partial['module.cells.18._op.conv3.adder'].cpu().detach().numpy())
weights.append(partial['module.cells.19._op.conv1.adder'].cpu().detach().numpy())
weights.append(partial['module.cells.19._op.conv2.adder'].cpu().detach().numpy())
weights.append(partial['module.cells.19._op.conv3.adder'].cpu().detach().numpy())
weights.append(partial['module.cells.20._op.conv1.adder'].cpu().detach().numpy())
weights.append(partial['module.cells.20._op.conv2.adder'].cpu().detach().numpy())
weights.append(partial['module.cells.20._op.conv3.adder'].cpu().detach().numpy())
# weights.append(shift_process(partial['module.cells.5._op.conv1.shift'],partial['module.cells.5._op.conv1.sign']).cpu().detach().numpy())
# weights.append(shift_process(partial['module.cells.5._op.conv2.shift'],partial['module.cells.5._op.conv2.sign']).cpu().detach().numpy())
# weights.append(shift_process(partial['module.cells.5._op.conv3.shift'],partial['module.cells.5._op.conv3.sign']).cpu().detach().numpy())
# weights.append(shift_process(partial['module.cells.11._op.conv1.shift'],partial['module.cells.11._op.conv1.sign']).cpu().detach().numpy())
# weights.append(shift_process(partial['module.cells.11._op.conv2.shift'],partial['module.cells.11._op.conv2.sign']).cpu().detach().numpy())
# weights.append(shift_process(partial['module.cells.11._op.conv3.shift'],partial['module.cells.11._op.conv3.sign']).cpu().detach().numpy())
# weights.append(partial['module.cells.12._op.conv1.adder'].cpu().detach().numpy())
# weights.append(partial['module.cells.12._op.conv2.adder'].cpu().detach().numpy())
# weights.append(partial['module.cells.12._op.conv3.adder'].cpu().detach().numpy())
# weights.append(partial['module.cells.13._op.conv1.weight'].cpu().detach().numpy())
# weights.append(partial['module.cells.13._op.conv2.weight'].cpu().detach().numpy())
# weights.append(partial['module.cells.13._op.conv3.weight'].cpu().detach().numpy())
# weights.append(partial['module.cells.19._op.conv1.weight'].cpu().detach().numpy())
# weights.append(partial['module.cells.19._op.conv2.weight'].cpu().detach().numpy())
# weights.append(partial['module.cells.19._op.conv3.weight'].cpu().detach().numpy())
# weights.append(shift_process(partial['module.cells.20._op.conv1.shift'],partial['module.cells.20._op.conv1.sign']).cpu().detach().numpy())
# weights.append(shift_process(partial['module.cells.20._op.conv2.shift'],partial['module.cells.20._op.conv2.sign']).cpu().detach().numpy())
# weights.append(shift_process(partial['module.cells.20._op.conv3.shift'],partial['module.cells.20._op.conv3.sign']).cpu().detach().numpy())

all_weights = np.zeros([0, ])
w_flatten = []
plot_weights = []
text = ['PW1', 'DW', 'PW2']
# x_range = [[-0.5,0.5],[-0.5,0.5],[-1,1],[-1,1],[-0.5,0.5],[-0.5,0.5],[-1,1],[-1,1],[-1,1],[-0.5,0.5],[-0.5,0.5]]
x_range = [[-0.05,0.05],[-0.05,0.05],[-0.5,0.5],[-0.5,0.5],[-0.05,0.05],[-0.05,0.05],[-0.5,0.5],[-0.5,0.5],[-0.5,0.5],[-0.05,0.05],[-0.05,0.05]]
for j in range(9):
    for i in range (3):
        w_flatten.append(np.reshape(weights[3*j+i], [-1]))
        plot_weights.append(np.concatenate([all_weights, w_flatten[3*j+i]], axis=0))
        ax[j][i].hist(plot_weights[3*j+i], bins=100, color="b",range=x_range[j])
        # ax[j][i].xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax[j][i].set_title(text[i])  
# plt.xticks(rotation=15)
plt.savefig("visualize/add_distribution.pdf")
plt.clf()
