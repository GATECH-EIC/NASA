import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import deepshift.ste as ste

def shift_process_ps(shift,sign):
    shift_clip = ste.clamp(shift, *((-1 * (2**(6 - 1) - 2), 0)))
    shift_rounded = ste.round(shift_clip, rounding='deterministic')
    # sign_rounded_signed = ste.sign(ste.round(sign, 'deterministic'))
    sign_rounded_signed = ste.sign(sign)
    weight_ps = ste.unsym_grad_mul(2**shift_rounded, sign_rounded_signed)
    return weight_ps
# weights.append(weight_ps)

def shift_process_q(weight):
    weight = ste.clampabs(weight.data, 2**(-1 * (2**(6 - 1) - 1)), 2**0)
    weight_q = ste.round_power_of_2(weight, 'deterministic')
    return weight_q


fig, ax = plt.subplots(9,3,figsize=(18, 24))
pretrained_model = torch.load('/Datadisk/shihh/NAS/baseline/CIFAR100_AddShift_q_scratch/weights_best_240.pth')
partial = pretrained_model['state_dict']
weights = []
shift= []
sign = []
weights.append(partial['module.cells.0._op.conv1.weight'].cpu().detach().numpy())
weights.append(partial['module.cells.0._op.conv2.weight'].cpu().detach().numpy())
weights.append(partial['module.cells.0._op.conv3.weight'].cpu().detach().numpy())
weights.append(partial['module.cells.4._op.conv1.weight'].cpu().detach().numpy())
weights.append(partial['module.cells.4._op.conv2.weight'].cpu().detach().numpy())
weights.append(partial['module.cells.4._op.conv3.weight'].cpu().detach().numpy())
weights.append(partial['module.cells.13._op.conv1.weight'].cpu().detach().numpy())
weights.append(partial['module.cells.13._op.conv2.weight'].cpu().detach().numpy())
weights.append(partial['module.cells.13._op.conv3.weight'].cpu().detach().numpy())
weights.append(shift_process(partial['module.cells.14._op.conv1.shift'],partial['module.cells.14._op.conv1.sign']).cpu().detach().numpy())
weights.append(shift_process(partial['module.cells.14._op.conv2.shift'],partial['module.cells.14._op.conv2.sign']).cpu().detach().numpy())
weights.append(shift_process(partial['module.cells.14._op.conv3.shift'],partial['module.cells.14._op.conv3.sign']).cpu().detach().numpy())
weights.append(shift_process(partial['module.cells.15._op.conv1.shift'],partial['module.cells.15._op.conv1.sign']).cpu().detach().numpy())
weights.append(shift_process(partial['module.cells.15._op.conv2.shift'],partial['module.cells.15._op.conv2.sign']).cpu().detach().numpy())
weights.append(shift_process(partial['module.cells.15._op.conv3.shift'],partial['module.cells.15._op.conv3.sign']).cpu().detach().numpy())
weights.append(shift_process(partial['module.cells.16._op.conv1.shift'],partial['module.cells.16._op.conv1.sign']).cpu().detach().numpy())
weights.append(shift_process(partial['module.cells.16._op.conv2.shift'],partial['module.cells.16._op.conv2.sign']).cpu().detach().numpy())
weights.append(shift_process(partial['module.cells.16._op.conv3.shift'],partial['module.cells.16._op.conv3.sign']).cpu().detach().numpy())
weights.append(partial['module.cells.17._op.conv1.weight'].cpu().detach().numpy())
weights.append(partial['module.cells.17._op.conv2.weight'].cpu().detach().numpy())
weights.append(partial['module.cells.17._op.conv3.weight'].cpu().detach().numpy())
weights.append(shift_process(partial['module.cells.18._op.conv1.shift'],partial['module.cells.18._op.conv1.sign']).cpu().detach().numpy())
weights.append(shift_process(partial['module.cells.18._op.conv2.shift'],partial['module.cells.18._op.conv2.sign']).cpu().detach().numpy())
weights.append(shift_process(partial['module.cells.18._op.conv3.shift'],partial['module.cells.18._op.conv3.sign']).cpu().detach().numpy())
weights.append(shift_process(partial['module.cells.20._op.conv1.shift'],partial['module.cells.20._op.conv1.sign']).cpu().detach().numpy())
weights.append(shift_process(partial['module.cells.20._op.conv2.shift'],partial['module.cells.20._op.conv2.sign']).cpu().detach().numpy())
weights.append(shift_process(partial['module.cells.20._op.conv3.shift'],partial['module.cells.20._op.conv3.sign']).cpu().detach().numpy())

all_weights = np.zeros([0, ])
w_flatten = []
plot_weights = []
text = ['PW1', 'DW', 'PW2']
x_range = [[-0.1,0.1],[-0.1,0.1],[-0.1,0.1],[0.001,0.001],[0.001,0.001],[0.001,0.001],[-0.1,0.1],[0.001,0.001],[0.001,0.001]]
for j in range(9):
    for i in range (3):
        w_flatten.append(np.reshape(weights[3*j+i], [-1]))
        plot_weights.append(np.concatenate([all_weights, w_flatten[3*j+i]], axis=0))
        ax[j][i].hist(plot_weights[3*j+i], bins=100, color="b",range=[-0.1,0.1])
        ax[j][i].xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax[j][i].set_title(text[i])  
# plt.xticks(rotation=15)
plt.savefig("100_shift_q_distribution.pdf")
plt.clf()
