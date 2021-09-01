import torch
from functools import partial
import sys
import numpy as np

def info(h, w, C_in, C_out, expansion, kernel_size=3, stride=1, padding=None, dilation=1, groups=1):
    h_out = h // stride
    w_out = w // stride

    conv_info = [[1, {'ch_out':[C_in*expansion,0],'ch_in':[C_in,0],'batch':[1,0],'col_out':[h,0],
                 'row_out':[w,0],'row_kernel':[1, 0],'col_kernel':[1,0]}],

                 [stride, {'ch_out':[C_in*expansion,0],'ch_in':[C_in*expansion,0],'batch':[1,0],'col_out':[h_out,0],
                 'row_out':[w_out,0],'row_kernel':[kernel_size,0],'col_kernel':[kernel_size,0]}],

                 [1, {'ch_out':[C_out,0],'ch_in':[C_in*expansion,0],'batch':[1,0],'col_out':[h_out,0],
                 'row_out':[w_out,0],'row_kernel':[1, 0],'col_kernel':[1 ,0]}]
                ]

    return conv_info


OPS = {
    'k3_e1' : partial(info, kernel_size=3, expansion=1, groups=1),
    'k3_e1_g2' : partial(info, kernel_size=3, expansion=1, groups=2),
    'k3_e3' : partial(info, kernel_size=3, expansion=3, groups=1),
    'k3_e6' : partial(info, kernel_size=3, expansion=6, groups=1),
    'k5_e1' : partial(info, kernel_size=5, expansion=1, groups=1),
    'k5_e1_g2' : partial(info, kernel_size=5, expansion=1, groups=2),
    'k5_e3' : partial(info, kernel_size=5, expansion=3, groups=1),
    'k5_e6' : partial(info, kernel_size=5, expansion=6, groups=1),
    'skip' : None
}


arch = torch.nn.functional.softmax(torch.load(sys.argv[1])['alpha'], dim=-1).argmax(-1).detach().cpu().numpy()

conv_info_sampled = []

h = w = 32
conv_info = []
conv_info.append([[1, {'ch_out':[16,0],'ch_in':[3,0],'batch':[1,0],'col_out':[32,0],
                 'row_out':[32,0],'row_kernel':[3, 0],'col_kernel':[3,0]}]])

layer = 0

for i in range(4):
    C_in = 16
    C_out = 16
    stride = 1

    choice = arch[layer]
    if choice == len(OPS) - 1:
        if C_in == C_out and stride == 1:
            continue
        else:
            conv_info.append([[1, {'ch_out':[C_out,0],'ch_in':[C_in,0],'batch':[1,0],'col_out':[h,0],
                            'row_out':[w,0],'row_kernel':[1, 0],'col_kernel':[1,0]}]])
    else:
        conv_info.append(OPS[list(OPS.keys())[choice]](h, w, C_in, C_out, stride=stride))

    layer += 1

for i in range(4):
    if i == 0:
        C_in = 16
        C_out = 32
        stride = 2

        choice = arch[layer]
        if choice == len(OPS) - 1:
            if C_in == C_out and stride == 1:
                continue
            else:
                conv_info.append([[1, {'ch_out':[C_out,0],'ch_in':[C_in,0],'batch':[1,0],'col_out':[h,0],
                                'row_out':[w,0],'row_kernel':[1, 0],'col_kernel':[1,0]}]])
        else:
            conv_info.append(OPS[list(OPS.keys())[choice]](h, w, C_in, C_out, stride=stride))

        h = h // 2
        w = w // 2

    else:
        C_in = 32
        C_out = 32
        stride = 1

        choice = arch[layer]
        if choice == len(OPS) - 1:
            if C_in == C_out and stride == 1:
                continue
            else:
                conv_info.append([[1, {'ch_out':[C_out,0],'ch_in':[C_in,0],'batch':[1,0],'col_out':[h,0],
                                'row_out':[w,0],'row_kernel':[1, 0],'col_kernel':[1,0]}]])
        else:
            conv_info.append(OPS[list(OPS.keys())[choice]](h, w, C_in, C_out, stride=stride))

    layer += 1

for i in range(4):
    if i == 0:
        C_in = 32
        C_out = 64
        stride = 2

        choice = arch[layer]
        if choice == len(OPS) - 1:
            if C_in == C_out and stride == 1:
                continue
            else:
                conv_info.append([[1, {'ch_out':[C_out,0],'ch_in':[C_in,0],'batch':[1,0],'col_out':[h,0],
                                'row_out':[w,0],'row_kernel':[1, 0],'col_kernel':[1,0]}]])
        else:
            conv_info.append(OPS[list(OPS.keys())[choice]](h, w, C_in, C_out, stride=stride))

        h = h // 2
        w = w // 2

    else:
        C_in = 64
        C_out = 64
        stride = 1

        choice = arch[layer]
        if choice == len(OPS) - 1:
            if C_in == C_out and stride == 1:
                continue
            else:
                conv_info.append([[1, {'ch_out':[C_out,0],'ch_in':[C_in,0],'batch':[1,0],'col_out':[h,0],
                                'row_out':[w,0],'row_kernel':[1, 0],'col_kernel':[1,0]}]])
        else:
            conv_info.append(OPS[list(OPS.keys())[choice]](h, w, C_in, C_out, stride=stride))

    layer += 1

conv_info.append([[1, {'ch_out':[128,0],'ch_in':[64,0],'batch':[1,0],'col_out':[8,0],
                 'row_out':[8,0],'row_kernel':[1, 0],'col_kernel':[1,0]}]])

conv_info_sampled.append(conv_info)

np.save('conv_info_final.npy', conv_info_sampled)

