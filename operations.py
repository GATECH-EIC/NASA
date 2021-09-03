from pdb import set_trace as bp
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from thop import profile
# from thop.count_hooks import count_convNd

import sys
import os.path as osp
from easydict import EasyDict as edict

from config_search import config
from adder import adder
from deepshift import modules
from deepshift import modules_q
from deepshift import utils as utils
from adder.quantize import quantize, quantize_grad, QuantMeasure, calculate_qparams


__all__ = ['ConvBlock', 'Skip','ConvNorm', 'OPS']

flops_lookup_table = {}
flops_file_name = osp.join(config.hw_platform_path, "flops_lookup_table.npy")
if osp.isfile(flops_file_name):
    flops_lookup_table = np.load(flops_file_name, allow_pickle=True).item()
print('config.hw_platform_path: ',config.hw_platform_path)

energy_lookup_table = {}
energy_file_name = osp.join(config.hw_platform_path, "energy_lookup_table.npy")
if osp.isfile(energy_file_name):
    energy_lookup_table = np.load(energy_file_name, allow_pickle=True).item()

latency_lookup_table = {}
latency_file_name = osp.join(config.hw_platform_path, "latency_lookup_table.npy")
if osp.isfile(latency_file_name):
    latency_lookup_table = np.load(latency_file_name, allow_pickle=True).item()

# TODO:
WEIGHT_BITS = 16

class QConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False,
                 ):
        super(QConv2d, self).__init__(in_channels, out_channels, kernel_size, stride=stride,
                                      padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.stride = stride
        self.padding = padding
        self.input_channels = in_channels
        self.output_channels = out_channels
        self.kernel_size = kernel_size
        self.groups = groups
        
        self.quantize_input_fw = QuantMeasure(shape_measure=(1, 1, 1, 1), flatten_dims=(1, -1), momentum=0.9)

        self.weight = torch.nn.Parameter(
            nn.init.normal_(torch.randn(
                out_channels,in_channels//groups,kernel_size,kernel_size)))

        # self.bias = bias
        # if bias:
        #     self.b = torch.nn.Parameter(
        #         nn.init.uniform_(torch.zeros(out_channels)))

    def forward(self, input):
        input_q = self.quantize_input_fw(input, WEIGHT_BITS)
        weight_qparams = calculate_qparams(self.weight, num_bits=WEIGHT_BITS, flatten_dims=(1, -1), reduce_dim=None)
        self.qweight = quantize(self.weight, qparams=weight_qparams)

        output = F.conv2d(input_q, self.qweight, None, self.stride,
                              self.padding, self.dilation, self.groups)
        
        # TODO:change to inference quantization
        output = quantize_grad(output, num_bits=WEIGHT_BITS, flatten_dims=(1, -1))
        # output = self.quantize_input_fw(output, WEIGHT_BITS)

        # if self.bias is not False : 
        #     output += self.b.unsqueeze(0).unsqueeze(2).unsqueeze(3) 
        #     # output = act_quant(output)

        return output

# TODO:
Conv2d = nn.Conv2d
# Conv2d = QConv2d
BatchNorm2d = nn.BatchNorm2d


class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        """Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,W] -> [N,C,H,W]"""
        N, C, H, W = x.size()
        g = self.groups
        assert (
            C % g == 0
        ), "Incompatible group size {} for input channel {}".format(g, C)
        return (
            x.view(N, g, int(C / g), H, W)
            .permute(0, 2, 1, 3, 4)
            .contiguous()
            .view(N, C, H, W)
        )


class ConvBlock(nn.Module):
    '''
    conv => norm => activation
    use native Conv2d, not slimmable
    '''
    def __init__(self, C_in, C_out,  layer_id, expansion=1, kernel_size=3, stride=1, padding=None, dilation=1, groups=1, bias=False):
        super(ConvBlock, self).__init__()
        self.C_in = C_in
        self.C_out = C_out

        self.layer_id = layer_id


        assert type(expansion) == int
        self.expansion = expansion
        self.kernel_size = kernel_size
        # print(self.kernel_size)
        assert stride in [1, 2]
        self.stride = stride
        if padding is None:
            # assume h_out = h_in / s
            self.padding = int(np.ceil((dilation * (kernel_size - 1) + 1 - stride) / 2.))
        else:
            self.padding = padding
        # print("padding=",padding)
        self.dilation = dilation
        assert type(groups) == int
        self.groups = groups
        self.bias = bias

        if self.groups > 1:
            self.shuffle = ChannelShuffle(self.groups)

        self.conv1 = Conv2d(C_in, C_in*expansion, kernel_size=1, stride=1, padding=0, dilation=1, groups=self.groups, bias=bias)
        self.bn1 = BatchNorm2d(C_in*expansion)

        self.conv2 = Conv2d(C_in*expansion, C_in*expansion, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=1, groups=C_in*expansion, bias=bias)
        self.bn2 = BatchNorm2d(C_in*expansion)
        self.conv3 = Conv2d(C_in*expansion, C_out, kernel_size=1, stride=1, padding=0, dilation=1, groups=self.groups, bias=bias)
        self.bn3 = BatchNorm2d(C_out)


        self.nl = nn.ReLU(inplace=True)


    # beta, mode, act_num, beta_param are for bit-widths search
    def forward(self, x):
        identity = x
        # print(x.size())
        x = self.nl(self.bn1(self.conv1(x)))
        # print(self.kernel_size)
        # print(self.padding)
        # print(x.size())

        if self.groups > 1:
            x = self.shuffle(x)

        x = self.nl(self.bn2(self.conv2(x)))
        # print(x.size())

        x = self.bn3(self.conv3(x))
        # print(x.size())
        if self.C_in == self.C_out and self.stride == 1:
            x += identity


        return x

    def set_stage(self, stage):
        assert stage == 'update_weight' or stage == 'update_arch'
        self.conv1.set_stage(stage)
        self.conv2.set_stage(stage)
        self.conv3.set_stage(stage)


    @staticmethod
    def _flops(h, w, C_in, C_out, expansion=1, kernel_size=3, stride=1, padding=None, dilation=1, groups=1, bias=False):
        layer_id = 1
        layer = ConvBlock(C_in, C_out, layer_id, expansion, kernel_size, stride, padding, dilation, groups, bias)
        flops, params = profile(layer, inputs=(torch.randn(1, C_in, h, w),))
        return flops
    

    def forward_energy(self, size):
        c_in, h_in, w_in = size

        assert c_in == self.C_in, "c_in %d, self.C_in %d"%(c_in, self.C_in)
        c_out = self.C_out

        if self.stride == 1:
            h_out = h_in; w_out = w_in
        else:
            h_out = h_in // 2; w_out = w_in // 2

        name = "ConvBlock_H%d_W%d_Cin%d_Cout%d_exp%d_kernel%d_stride%d_group%d"%(h_in, w_in, c_in, c_out, self.expansion, self.kernel_size, self.stride, self.groups)

        if name in energy_lookup_table:
            energy = energy_lookup_table[name]
        else:
            print("not found in energy_lookup_table:", name)
            sys.exit()
        return energy, (c_out, h_out, w_out)


    def forward_latency(self, size):
        c_in, h_in, w_in = size

        assert c_in == self.C_in, "c_in %d, self.C_in %d"%(c_in, self.C_in)
        c_out = self.C_out

        if self.stride == 1:
            h_out = h_in; w_out = w_in
        else:
            h_out = h_in // 2; w_out = w_in // 2

        name = "ConvBlock_H%d_W%d_Cin%d_Cout%d_exp%d_kernel%d_stride%d_group%d"%(h_in, w_in, c_in, c_out, self.expansion, self.kernel_size, self.stride, self.groups)

        if name in latency_lookup_table:
            latency = latency_lookup_table[name]
        else:
            print("not found in latency_lookup_table:", name)
            sys.exit()

        return latency, (c_out, h_out, w_out)


    def forward_flops(self, size):
        c_in, h_in, w_in = size

        assert c_in == self.C_in, "c_in %d, self.C_in %d"%(c_in, self.C_in)
        c_out = self.C_out

        if self.stride == 1:
            h_out = h_in; w_out = w_in
        else:
            h_out = h_in // 2; w_out = w_in // 2

        name = "ConvBlock_H%d_W%d_Cin%d_Cout%d_exp%d_kernel%d_stride%d_group%d"%(h_in, w_in, c_in, c_out, self.expansion, self.kernel_size, self.stride, self.groups)
        if name in flops_lookup_table:
            flops = flops_lookup_table[name]
        else:
            print("not found in flops_lookup_table:", name)
            flops = ConvBlock._flops(h_in, w_in, c_in, c_out, self.expansion, self.kernel_size, self.stride, self.padding, self.dilation, self.groups, self.bias)
            flops_lookup_table[name] = flops
            np.save(flops_file_name, flops_lookup_table)

        return flops, (c_out, h_out, w_out)


    def layer_info(self, size):
        c_in, h_in, w_in = size

        assert c_in == self.C_in, "c_in %d, self.C_in %d"%(c_in, self.C_in)
        c_out = self.C_out

        if self.stride == 1:
            h_out = h_in; w_out = w_in
        else:
            h_out = h_in // 2; w_out = w_in // 2

        if self.groups == 2:
            group_type = 2
            group = 2
        else:
            group_type = 0
            group = 1

        block_info = [
                     [1, {'ch_out':[self.C_in*self.expansion/group,0],'ch_in':[self.C_in/group,0],'batch':[1,0],'col_out':[h_in,0],
                     'row_out':[w_in,0],'row_kernel':[1, 0],'col_kernel':[1,0]}, group_type, group],

                     [self.stride, {'ch_out':[self.C_in*self.expansion,0],'batch':[1,0],'col_out':[h_out,0],
                     'row_out':[w_out,0],'row_kernel':[self.kernel_size,0],'col_kernel':[self.kernel_size,0]}, 1, 1],

                     [1, {'ch_out':[self.C_out/group,0],'ch_in':[self.C_in*self.expansion/group,0],'batch':[1,0],'col_out':[h_out,0],
                     'row_out':[w_out,0],'row_kernel':[1, 0],'col_kernel':[1 ,0]}, group_type, group]
                     ]

        return block_info, (self.C_out, h_out, w_out)


class Skip(nn.Module):
    def __init__(self, C_in, C_out, layer_id, stride=1):
        super(Skip, self).__init__()
        assert stride in [1, 2]
        assert C_out % 2 == 0, 'C_out=%d'%C_out
        self.C_in = C_in
        self.C_out = C_out
        self.stride = stride

        self.layer_id = layer_id


        self.kernel_size = 1
        self.padding = 0

        if stride == 2 or C_in != C_out:
            self.conv = Conv2d(C_in, C_out, 1, stride=stride, padding=0, bias=False)
            self.bn = BatchNorm2d(C_out)
            self.relu = nn.ReLU(inplace=True)

    @staticmethod
    def _flops(h, w, C_in, C_out, stride=1):
        layer = Skip(C_in, C_out, stride)
        flops, params = profile(layer, inputs=(torch.randn(1, C_in, h, w),))
        return flops

    def forward_energy(self, size):
        c_in, h_in, w_in = size

        assert c_in == self.C_in
        c_out = self.C_out

        if self.stride == 1:
            h_out = h_in; w_out = w_in
        else:
            h_out = h_in // 2; w_out = w_in // 2

        name = "Skip_H%d_W%d_Cin%d_Cout%d_stride%d"%(h_in, w_in, c_in, c_out, self.stride)

        if name in energy_lookup_table:
            energy = energy_lookup_table[name]
        else:
            print("not found in energy_lookup_table:", name)
            sys.exit()
        return energy, (c_out, h_out, w_out)


    def forward_latency(self, size):
        c_in, h_in, w_in = size

        assert c_in == self.C_in
        c_out = self.C_out

        if self.stride == 1:
            h_out = h_in; w_out = w_in
        else:
            h_out = h_in // 2; w_out = w_in // 2

        name = "Skip_H%d_W%d_Cin%d_Cout%d_stride%d"%(h_in, w_in, c_in, c_out, self.stride)

        if name in latency_lookup_table:
            latency = latency_lookup_table[name]
        else:
            print("not found in latency_lookup_table:", name)
            sys.exit()
        return latency, (c_out, h_out, w_out)


    def forward_flops(self, size):
        c_in, h_in, w_in = size

        assert c_in == self.C_in, "c_in %d, self.C_in %d"%(c_in, self.C_in)
        c_out = self.C_out

        if self.stride == 1:
            h_out = h_in; w_out = w_in
        else:
            h_out = h_in // 2; w_out = w_in // 2

        name = "Skip_H%d_W%d_Cin%d_Cout%d_stride%d"%(h_in, w_in, c_in, c_out, self.stride)

        if name in flops_lookup_table:
            flops = flops_lookup_table[name]
        else:
            print("not found in flops_lookup_table:", name)
            flops = Skip._flops(h_in, w_in, c_in, c_out, self.stride)
            flops_lookup_table[name] = flops
            np.save(flops_file_name, flops_lookup_table)

        return flops, (c_out, h_out, w_out)


    def forward(self, x):
        if hasattr(self, 'conv'):
            out = self.conv(x)
            out = self.bn(out)
            out = self.relu(out)
        else:
            out = x


        return out



    def set_stage(self, stage):
        if hasattr(self, 'conv'):
            assert stage == 'update_weight' or stage == 'update_arch'
            self.conv.set_stage(stage)


    def layer_info(self, size):
        c_in, h_in, w_in = size

        assert c_in == self.C_in, "c_in %d, self.C_in %d"%(c_in, self.C_in)
        c_out = self.C_out

        if self.stride == 1:
            h_out = h_in; w_out = w_in
        else:
            h_out = h_in // 2; w_out = w_in // 2

        if self.stride == 2 or self.C_in != self.C_out:
            conv_info = [
                        [self.stride, {'ch_out':[self.C_out,0],'ch_in':[self.C_in,0],'batch':[1,0],'col_out':[h_out,0],'row_out':[w_out,0],
                        'row_kernel':[1,0],'col_kernel':[1,0]}, 0, 1]
                        ]
        else:
            conv_info = []

        return conv_info, (self.C_out, h_out, w_out)


def Shiftlayer(in_planes, out_planes, kernel_size=1, stride=1, padding=1, groups=1, bias=False, freeze_sign = False, use_kernel=False, use_cuda=True, shift_type='Q',
    rounding='deterministic', weight_bits=6, sign_threshold_ps=None, quant_bits=16):
    # conversion_count = 0

    if shift_type == 'Q':
        shift_conv2d = modules_q.Conv2dShiftQ(in_channels=in_planes, out_channels=out_planes, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=bias, use_kernel=use_kernel, use_cuda=use_cuda, rounding=rounding, weight_bits=weight_bits,quant_bits=quant_bits)
    #     shift_conv2d.weight = conv2d.weight
    #     if conv2d.bias is not None:
    #         shift_conv2d.bias.data = utils.round_to_fixed(conv2d.bias, fraction=16, integer=16)

    #     if use_cuda==True and use_kernel == True:
    #         shift_conv2d.conc_weight = utils.compress_bits(*utils.get_shift_and_sign(conv2d.weight))

    elif shift_type == 'PS':
        shift_conv2d = modules.Conv2dShift(in_channels=in_planes, out_channels=out_planes, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=bias, freeze_sign=freeze_sign, use_kernel=use_kernel, use_cuda=use_cuda,
                                        rounding=rounding, weight_bits=weight_bits, threshold=sign_threshold_ps, quant_bits=quant_bits)


    # model._modules[name] = shift_conv2d
    # conversion_count += 1

    return shift_conv2d

# TODO:
def Addlayer(in_planes, out_planes, kernel_size=1, stride=1, padding=1, groups=1, quantize=False, weight_bits=WEIGHT_BITS, sparsity=0, quantize_v='sbm'):
    " 3x3 convolution with padding "
    return adder.Adder2D(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False,
                         quantize=quantize, weight_bits=weight_bits, sparsity=sparsity, quantize_v=quantize_v)
    # shift = shift_conv(in_planes, out_planes, kernel_size, stride=stride, padding=padding, bias=False)
    # return shift


def ShiftAddlayer(in_planes, out_planes, kernel_size=1, stride=1, padding=1, groups=1, quantize=True, weight_bits=6, sparsity=0, quantize_v='sbm'):
    " 3x3 convolution with padding "
    # add = Addlayer(in_planes, out_planes, kernel_size, stride=1, padding=padding, groups=groups)
    shift = Shiftlayer(in_planes, out_planes, kernel_size, stride=stride, padding=padding, groups=groups, weight_bits=weight_bits)
    add = Addlayer(out_planes, out_planes, kernel_size, stride=1, padding=padding, groups=groups)
    return nn.Sequential(shift, add)
    # return nn.Sequential(add, shift)
    # return add


class ShiftBlock(nn.Module):
    '''
    conv => norm => activation
    use native Conv2d, not slimmable
    '''
    def __init__(self, C_in, C_out, layer_id, expansion=1, kernel_size=3, stride=1, padding=None, dilation=1, groups=1, bias=False):
        super(ShiftBlock, self).__init__()
        self.C_in = C_in
        self.C_out = C_out

        self.layer_id = layer_id

        assert type(expansion) == int
        self.expansion = expansion
        self.kernel_size = kernel_size
        # print(self.kernel_size)
        assert stride in [1, 2]
        self.stride = stride
        if padding is None:
            # assume h_out = h_in / s
            self.padding = int(np.ceil((dilation * (kernel_size - 1) + 1 - stride) / 2.))
        else:
            self.padding = padding
        # print("padding=",padding)
        self.dilation = dilation
        assert type(groups) == int
        self.groups = groups
        self.bias = bias

        if self.groups > 1:
            self.shuffle = ChannelShuffle(self.groups)

        self.conv1 = Shiftlayer(C_in, C_in*expansion, kernel_size=1, stride=1, padding=0, groups=self.groups)
        self.bn1 = BatchNorm2d(C_in*expansion)

        self.conv2 = Shiftlayer(C_in*expansion, C_in*expansion, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, groups=C_in*expansion)
        self.bn2 = BatchNorm2d(C_in*expansion)
        self.conv3 = Shiftlayer(C_in*expansion, C_out, kernel_size=1, stride=1, padding=0, groups=self.groups)
        self.bn3 = BatchNorm2d(C_out)


        self.nl = nn.ReLU(inplace=True)


    # beta, mode, act_num, beta_param are for bit-widths search
    def forward(self, x):
        identity = x
        # print(x.size())
        x = self.nl(self.bn1(self.conv1(x)))
        # print(self.kernel_size)
        # print(self.padding)
        # print(x.size())

        if self.groups > 1:
            x = self.shuffle(x)

        x = self.nl(self.bn2(self.conv2(x)))
        # print(x.size())

        x = self.bn3(self.conv3(x))
        # print(x.size())
        if self.C_in == self.C_out and self.stride == 1:
            x += identity

        return x

    def set_stage(self, stage):
        assert stage == 'update_weight' or stage == 'update_arch'
        self.conv1.set_stage(stage)
        self.conv2.set_stage(stage)
        self.conv3.set_stage(stage)

    @staticmethod
    def layer_info(self, size):
        c_in, h_in, w_in = size

        assert c_in == self.C_in, "c_in %d, self.C_in %d"%(c_in, self.C_in)
        c_out = self.C_out

        if self.stride == 1:
            h_out = h_in; w_out = w_in
        else:
            h_out = h_in // 2; w_out = w_in // 2

        if self.groups == 2:
            group_type = 2
            group = 2
        else:
            group_type = 0
            group = 1

        block_info = [
                     [1, {'ch_out':[self.C_in*self.expansion/group,0],'ch_in':[self.C_in/group,0],'batch':[1,0],'col_out':[h_in,0],
                     'row_out':[w_in,0],'row_kernel':[1, 0],'col_kernel':[1,0]}, group_type, group],

                     [self.stride, {'ch_out':[self.C_in*self.expansion,0],'batch':[1,0],'col_out':[h_out,0],
                     'row_out':[w_out,0],'row_kernel':[self.kernel_size,0],'col_kernel':[self.kernel_size,0]}, 1, 1],

                     [1, {'ch_out':[self.C_out/group,0],'ch_in':[self.C_in*self.expansion/group,0],'batch':[1,0],'col_out':[h_out,0],
                     'row_out':[w_out,0],'row_kernel':[1, 0],'col_kernel':[1 ,0]}, group_type, group]
                     ]

        return block_info, (self.C_out, h_out, w_out)

    def forward_flops(self, size):
        c_in, h_in, w_in = size

        assert c_in == self.C_in, "c_in %d, self.C_in %d"%(c_in, self.C_in)
        c_out = self.C_out

        if self.stride == 1:
            h_out = h_in; w_out = w_in
        else:
            h_out = h_in // 2; w_out = w_in // 2

        name = "ConvBlock_H%d_W%d_Cin%d_Cout%d_exp%d_kernel%d_stride%d_group%d"%(h_in, w_in, c_in, c_out, self.expansion, self.kernel_size, self.stride, self.groups)
        if name in flops_lookup_table:
            flops = flops_lookup_table[name]
        else:
            print("not found in flops_lookup_table:", name)
            flops = ConvBlock._flops(h_in, w_in, c_in, c_out, self.expansion, self.kernel_size, self.stride, self.padding, self.dilation, self.groups, self.bias)
            flops_lookup_table[name] = flops
            np.save(flops_file_name, flops_lookup_table)

        return flops/8, (c_out, h_out, w_out)


class AddBlock(nn.Module):
    '''
    conv => norm => activation
    use native Conv2d, not slimmable
    '''
    def __init__(self, C_in, C_out, layer_id, expansion=1, kernel_size=3, stride=1, padding=None, dilation=1, groups=1, bias=False):
        super(AddBlock, self).__init__()
        self.C_in = C_in
        self.C_out = C_out

        self.layer_id = layer_id

        assert type(expansion) == int
        self.expansion = expansion
        self.kernel_size = kernel_size
        # print(self.kernel_size)
        assert stride in [1, 2]
        self.stride = stride
        if padding is None:
            # assume h_out = h_in / s
            self.padding = int(np.ceil((dilation * (kernel_size - 1) + 1 - stride) / 2.))
        else:
            self.padding = padding
        # print("padding=",padding)
        self.dilation = dilation
        assert type(groups) == int
        self.groups = groups
        self.bias = bias

        if self.groups > 1:
            self.shuffle = ChannelShuffle(self.groups)

        self.conv1 = Addlayer(C_in, C_in*expansion, kernel_size=1, stride=1, padding=0, groups=self.groups)
        self.bn1 = BatchNorm2d(C_in*expansion)

        self.conv2 = Addlayer(C_in*expansion, C_in*expansion, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, groups=C_in*expansion)
        self.bn2 = BatchNorm2d(C_in*expansion)
        self.conv3 = Addlayer(C_in*expansion, C_out, kernel_size=1, stride=1, padding=0, groups=self.groups)
        self.bn3 = BatchNorm2d(C_out)


        self.nl = nn.ReLU(inplace=True)


    # beta, mode, act_num, beta_param are for bit-widths search
    def forward(self, x):
        identity = x
        # print(x.size())
        x = self.nl(self.bn1(self.conv1(x)))
        # print(self.kernel_size)
        # print(self.padding)
        # print(x.size())

        if self.groups > 1:
            x = self.shuffle(x)

        x = self.nl(self.bn2(self.conv2(x)))
        # print(x.size())

        x = self.bn3(self.conv3(x))
        # print(x.size())
        if self.C_in == self.C_out and self.stride == 1:
            x += identity

        return x

    def set_stage(self, stage):
        assert stage == 'update_weight' or stage == 'update_arch'
        self.conv1.set_stage(stage)
        self.conv2.set_stage(stage)
        self.conv3.set_stage(stage)

    @staticmethod
    def layer_info(self, size):
        c_in, h_in, w_in = size

        assert c_in == self.C_in, "c_in %d, self.C_in %d"%(c_in, self.C_in)
        c_out = self.C_out

        if self.stride == 1:
            h_out = h_in; w_out = w_in
        else:
            h_out = h_in // 2; w_out = w_in // 2

        if self.groups == 2:
            group_type = 2
            group = 2
        else:
            group_type = 0
            group = 1

        block_info = [
                     [1, {'ch_out':[self.C_in*self.expansion/group,0],'ch_in':[self.C_in/group,0],'batch':[1,0],'col_out':[h_in,0],
                     'row_out':[w_in,0],'row_kernel':[1, 0],'col_kernel':[1,0]}, group_type, group],

                     [self.stride, {'ch_out':[self.C_in*self.expansion,0],'batch':[1,0],'col_out':[h_out,0],
                     'row_out':[w_out,0],'row_kernel':[self.kernel_size,0],'col_kernel':[self.kernel_size,0]}, 1, 1],

                     [1, {'ch_out':[self.C_out/group,0],'ch_in':[self.C_in*self.expansion/group,0],'batch':[1,0],'col_out':[h_out,0],
                     'row_out':[w_out,0],'row_kernel':[1, 0],'col_kernel':[1 ,0]}, group_type, group]
                     ]

        return block_info, (self.C_out, h_out, w_out)

    def forward_flops(self, size):
        c_in, h_in, w_in = size

        assert c_in == self.C_in, "c_in %d, self.C_in %d"%(c_in, self.C_in)
        c_out = self.C_out

        if self.stride == 1:
            h_out = h_in; w_out = w_in
        else:
            h_out = h_in // 2; w_out = w_in // 2

        name = "ConvBlock_H%d_W%d_Cin%d_Cout%d_exp%d_kernel%d_stride%d_group%d"%(h_in, w_in, c_in, c_out, self.expansion, self.kernel_size, self.stride, self.groups)
        if name in flops_lookup_table:
            flops = flops_lookup_table[name]
        else:
            print("not found in flops_lookup_table:", name)
            flops = ConvBlock._flops(h_in, w_in, c_in, c_out, self.expansion, self.kernel_size, self.stride, self.padding, self.dilation, self.groups, self.bias)
            flops_lookup_table[name] = flops
            np.save(flops_file_name, flops_lookup_table)

        return flops/5, (c_out, h_out, w_out)


class ShiftAddBlock(nn.Module):
    '''
    conv => norm => activation
    use native Conv2d, not slimmable
    '''
    def __init__(self, C_in, C_out, layer_id, expansion=1, kernel_size=3, stride=1, padding=None, dilation=1, groups=1, bias=False):
        super(ShiftAddBlock, self).__init__()
        self.C_in = C_in
        self.C_out = C_out

        self.layer_id = layer_id

        assert type(expansion) == int
        self.expansion = expansion
        self.kernel_size = kernel_size
        # print(self.kernel_size)
        assert stride in [1, 2]
        self.stride = stride
        if padding is None:
            # assume h_out = h_in / s
            self.padding = int(np.ceil((dilation * (kernel_size - 1) + 1 - stride) / 2.))
        else:
            self.padding = padding
        # print("padding=",padding)
        self.dilation = dilation
        assert type(groups) == int
        self.groups = groups
        self.bias = bias

        if self.groups > 1:
            self.shuffle = ChannelShuffle(self.groups)

        self.conv1 = ShiftAddlayer(C_in, C_in*expansion, kernel_size=1, stride=1, padding=0, groups=self.groups)
        self.bn1 = BatchNorm2d(C_in*expansion)

        self.conv2 = ShiftAddlayer(C_in*expansion, C_in*expansion, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, groups=C_in*expansion)
        self.bn2 = BatchNorm2d(C_in*expansion)
        self.conv3 = ShiftAddlayer(C_in*expansion, C_out, kernel_size=1, stride=1, padding=0, groups=self.groups)
        self.bn3 = BatchNorm2d(C_out)


        self.nl = nn.ReLU(inplace=True)


    # beta, mode, act_num, beta_param are for bit-widths search
    def forward(self, x):
        identity = x
        # print(x.size())
        x = self.nl(self.bn1(self.conv1(x)))
        # if (self.layer_id == 3):
        #     print("input:",identity)
        #     print("weight:",self.conv1.weight)
        #     print("output:",self.conv1(x))
        # print(self.kernel_size)
        # print(self.padding)
        # print(x.size())

        if self.groups > 1:
            x = self.shuffle(x)

        x = self.nl(self.bn2(self.conv2(x)))
        # print(x.size())

        x = self.bn3(self.conv3(x))
        # print(x.size())
        if self.C_in == self.C_out and self.stride == 1:
            x += identity

        return x

    def set_stage(self, stage):
        assert stage == 'update_weight' or stage == 'update_arch'
        self.conv1.set_stage(stage)
        self.conv2.set_stage(stage)
        self.conv3.set_stage(stage)

    @staticmethod
    def layer_info(self, size):
        c_in, h_in, w_in = size

        assert c_in == self.C_in, "c_in %d, self.C_in %d"%(c_in, self.C_in)
        c_out = self.C_out

        if self.stride == 1:
            h_out = h_in; w_out = w_in
        else:
            h_out = h_in // 2; w_out = w_in // 2

        if self.groups == 2:
            group_type = 2
            group = 2
        else:
            group_type = 0
            group = 1

        block_info = [
                     [1, {'ch_out':[self.C_in*self.expansion/group,0],'ch_in':[self.C_in/group,0],'batch':[1,0],'col_out':[h_in,0],
                     'row_out':[w_in,0],'row_kernel':[1, 0],'col_kernel':[1,0]}, group_type, group],

                     [self.stride, {'ch_out':[self.C_in*self.expansion,0],'batch':[1,0],'col_out':[h_out,0],
                     'row_out':[w_out,0],'row_kernel':[self.kernel_size,0],'col_kernel':[self.kernel_size,0]}, 1, 1],

                     [1, {'ch_out':[self.C_out/group,0],'ch_in':[self.C_in*self.expansion/group,0],'batch':[1,0],'col_out':[h_out,0],
                     'row_out':[w_out,0],'row_kernel':[1, 0],'col_kernel':[1 ,0]}, group_type, group]
                     ]

        return block_info, (self.C_out, h_out, w_out)

    def forward_flops(self, size):
        c_in, h_in, w_in = size
        assert c_in == self.C_in, "c_in %d, self.C_in %d"%(c_in, self.C_in)
        c_out = self.C_out

        if self.stride == 1:
            h_out = h_in; w_out = w_in
        else:
            h_out = h_in // 2; w_out = w_in // 2

        name = "ShiftAddBlock_H%d_W%d_Cin%d_Cout%d_exp%d_kernel%d_stride%d_group%d"%(h_in, w_in, c_in, c_out, self.expansion, self.kernel_size, self.stride, self.groups)
        if name in flops_lookup_table:
            flops = flops_lookup_table[name]
        else:
            print("not found in flops_lookup_table:", name)
            
            # ############### shift layer flops #########
            name_shift = "ConvBlock_H%d_W%d_Cin%d_Cout%d_exp%d_kernel%d_stride%d_group%d"%(h_in, w_in, c_in, c_out, self.expansion, self.kernel_size, self.stride, self.groups)
            if name_shift in flops_lookup_table:
                flops_shift = flops_lookup_table[name_shift]
            else:
                flops_shift = ConvBlock._flops(h_in, w_in, c_in, c_out, self.expansion, self.kernel_size, self.stride, self.padding, self.dilation, self.groups, self.bias)
                flops_lookup_table[name_shift] = flops_shift
            # np.save(flops_file_name, flops_lookup_table)

            # ############### add layer flops #########
            layer = Addlayer(c_in*self.expansion, c_in*self.expansion, kernel_size=1, stride=1, padding=0)
            flop, params = profile(layer, inputs=(torch.randn(1, c_in*self.expansion, h_in, w_in),))
            flops_add = flop
            layer = Addlayer(c_in*self.expansion, c_in*self.expansion, kernel_size=self.kernel_size, stride=1, padding=self.padding)
            flop, params = profile(layer, inputs=(torch.randn(1, c_in*self.expansion, h_out, w_out),))
            flops_add += flop
            layer = Addlayer(c_out, c_out, kernel_size=self.kernel_size, stride=1, padding=self.padding)
            flop, params = profile(layer, inputs=(torch.randn(1, c_out, h_out, w_out),))
            flops_add += flop
            flops = flops_add/5 + flops_shift/8
            
            flops_lookup_table[name] = flops
            np.save(flops_file_name, flops_lookup_table)

        return flops, (c_out, h_out, w_out)


class ConvNorm(nn.Module):
    '''
    conv => norm => activation
    use native Conv2d, not slimmable
    '''
    def __init__(self, C_in, C_out, kernel_size=3, stride=1, padding=None, dilation=1, groups=1, bias=False):
        super(ConvNorm, self).__init__()
        self.C_in = C_in
        self.C_out = C_out


        self.kernel_size = kernel_size
        assert stride in [1, 2]
        self.stride = stride
        if padding is None:
            # assume h_out = h_in / s
            self.padding = int(np.ceil((dilation * (kernel_size - 1) + 1 - stride) / 2.))
        else:
            self.padding = padding
        self.dilation = dilation
        assert type(groups) == int
        self.groups = groups
        self.bias = bias

        self.conv = Conv2d(C_in, C_out, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, 
                            dilation=self.dilation, groups=self.groups, bias=bias)
        self.bn = BatchNorm2d(C_out)

        self.relu = nn.ReLU(inplace=True)



    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))


        return x



    def set_stage(self, stage):
        assert stage == 'update_weight' or stage == 'update_arch'
        self.conv.set_stage(stage)


    @staticmethod
    def _flops(h, w, C_in, C_out, kernel_size=3, stride=1, padding=None, dilation=1, groups=1, bias=False):
        layer = ConvNorm(C_in, C_out, kernel_size, stride, padding, dilation, groups, bias)
        flops, params = profile(layer, inputs=(torch.randn(1, C_in, h, w),))
        return flops
    

    def forward_energy(self, size):
        c_in, h_in, w_in = size

        assert c_in == self.C_in, "c_in %d, self.C_in %d"%(c_in, self.C_in)
        c_out = self.C_out

        if self.stride == 1:
            h_out = h_in; w_out = w_in
        else:
            h_out = h_in // 2; w_out = w_in // 2

        name = "ConvNorm_H%d_W%d_Cin%d_Cout%d_kernel%d_stride%d_group%d"%(h_in, w_in, c_in, c_out, self.kernel_size, self.stride, self.groups)

        if name in energy_lookup_table:
            energy = energy_lookup_table[name]
        else:
            print("not found in energy_lookup_table:", name)
            sys.exit()
        return energy, (c_out, h_out, w_out)


    def forward_latency(self, size):
        c_in, h_in, w_in = size

        assert c_in == self.C_in, "c_in %d, self.C_in %d"%(c_in, self.C_in)
        c_out = self.C_out

        if self.stride == 1:
            h_out = h_in; w_out = w_in
        else:
            h_out = h_in // 2; w_out = w_in // 2

        name = "ConvNorm_H%d_W%d_Cin%d_Cout%d_kernel%d_stride%d_group%d"%(h_in, w_in, c_in, c_out, self.kernel_size, self.stride, self.groups)

        if name in latency_lookup_table:
            latency = latency_lookup_table[name]
        else:
            print("not found in latency_lookup_table:", name)
            sys.exit()
        return latency, (c_out, h_out, w_out)


    def forward_flops(self, size):
        c_in, h_in, w_in = size

        assert c_in == self.C_in, "c_in %d, self.C_in %d"%(c_in, self.C_in)
        c_out = self.C_out

        if self.stride == 1:
            h_out = h_in; w_out = w_in
        else:
            h_out = h_in // 2; w_out = w_in // 2

        name = "ConvNorm_H%d_W%d_Cin%d_Cout%d_kernel%d_stride%d_group%d"%(h_in, w_in, c_in, c_out, self.kernel_size, self.stride, self.groups)
        if name in flops_lookup_table:
            flops = flops_lookup_table[name]
        else:
            print("not found in flops_lookup_table:", name)
            flops = ConvNorm._flops(h_in, w_in, c_in, c_out, self.kernel_size, self.stride, self.padding, self.dilation, self.groups, self.bias)
            flops_lookup_table[name] = flops
            np.save(flops_file_name, flops_lookup_table)

        return flops, (c_out, h_out, w_out)


    def layer_info(self, size):
        c_in, h_in, w_in = size

        assert c_in == self.C_in, "c_in %d, self.C_in %d"%(c_in, self.C_in)
        c_out = self.C_out

        if self.stride == 1:
            h_out = h_in; w_out = w_in
        else:
            h_out = h_in // 2; w_out = w_in // 2

        conv_info = [
                    [self.stride, {'ch_out':[self.C_out,0],'ch_in':[self.C_in,0],'batch':[1,0],'col_out':[h_out,0],
                    'row_out':[w_out,0],'row_kernel':[self.kernel_size, 0],'col_kernel':[self.kernel_size,0]}, 0, 1]
                    ]
            
        return conv_info, (self.C_out, h_out, w_out)



# OPS = {
#     'k3_e1' : lambda C_in, C_out, layer_id, stride: ConvBlock(C_in, C_out, layer_id, expansion=1, kernel_size=3, stride=stride, groups=1),
#     'k3_e1_g2' : lambda C_in, C_out, layer_id, stride: ConvBlock(C_in, C_out, layer_id, expansion=1, kernel_size=3, stride=stride, groups=2),
#     'k3_e3' : lambda C_in, C_out, layer_id, stride: ConvBlock(C_in, C_out, layer_id, expansion=3, kernel_size=3, stride=stride, groups=1),
#     'k3_e6' : lambda C_in, C_out, layer_id, stride: ConvBlock(C_in, C_out, layer_id, expansion=6, kernel_size=3, stride=stride, groups=1),
#     'k5_e1' : lambda C_in, C_out, layer_id, stride: ConvBlock(C_in, C_out, layer_id, expansion=1, kernel_size=5, stride=stride, groups=1),
#     'k5_e1_g2' : lambda C_in, C_out, layer_id, stride: ConvBlock(C_in, C_out, layer_id, expansion=1, kernel_size=5, stride=stride, groups=2),
#     'k5_e3' : lambda C_in, C_out, layer_id, stride: ConvBlock(C_in, C_out, layer_id, expansion=3, kernel_size=5, stride=stride, groups=1),
#     'k5_e6' : lambda C_in, C_out, layer_id, stride: ConvBlock(C_in, C_out, layer_id, expansion=6, kernel_size=5, stride=stride, groups=1),
#     'skip' : lambda C_in, C_out, layer_id, stride: Skip(C_in, C_out, layer_id, stride),
#     'k3_shift' : lambda C_in, C_out, layer_id, stride: Shift(C_in, C_out, layer_id, kernel_size=3, stride=stride, groups=1),
#     'k3_add' : lambda C_in, C_out, layer_id, stride: Add(C_in, C_out, layer_id, kernel_size=3, stride=stride, groups=1),
#     'k3_shiftadd' : lambda C_in, C_out, layer_id, stride: ShiftAdd(C_in, C_out, layer_id, kernel_size=3, stride=stride, groups=1),
#     'k5_shift' : lambda C_in, C_out, layer_id, stride: Shift(C_in, C_out, layer_id, kernel_size=5, stride=stride, groups=1),
#     'k5_add' : lambda C_in, C_out, layer_id, stride: Add(C_in, C_out, layer_id, kernel_size=5, stride=stride, groups=1),
#     'k5_shiftadd' : lambda C_in, C_out, layer_id, stride: ShiftAdd(C_in, C_out, layer_id, kernel_size=5, stride=stride, groups=1)
# }
OPS = {
    'k3_e1' : lambda C_in, C_out, layer_id, stride: ConvBlock(C_in, C_out, layer_id, expansion=1, kernel_size=3, stride=stride, groups=1),
    'k3_e1_g2' : lambda C_in, C_out, layer_id, stride: ConvBlock(C_in, C_out, layer_id, expansion=1, kernel_size=3, stride=stride, groups=2),
    'k3_e3' : lambda C_in, C_out, layer_id, stride: ConvBlock(C_in, C_out, layer_id, expansion=3, kernel_size=3, stride=stride, groups=1),
    'k3_e6' : lambda C_in, C_out, layer_id, stride: ConvBlock(C_in, C_out, layer_id, expansion=6, kernel_size=3, stride=stride, groups=1),
    'k5_e1' : lambda C_in, C_out, layer_id, stride: ConvBlock(C_in, C_out, layer_id, expansion=1, kernel_size=5, stride=stride, groups=1),
    'k5_e1_g2' : lambda C_in, C_out, layer_id, stride: ConvBlock(C_in, C_out, layer_id, expansion=1, kernel_size=5, stride=stride, groups=2),
    'k5_e3' : lambda C_in, C_out, layer_id, stride: ConvBlock(C_in, C_out, layer_id, expansion=3, kernel_size=5, stride=stride, groups=1),
    'k5_e6' : lambda C_in, C_out, layer_id, stride: ConvBlock(C_in, C_out, layer_id, expansion=6, kernel_size=5, stride=stride, groups=1),
    'skip' : lambda C_in, C_out, layer_id, stride: Skip(C_in, C_out, layer_id, stride),
    'add_k3_e1' : lambda C_in, C_out, layer_id, stride: AddBlock(C_in, C_out, layer_id, expansion=1, kernel_size=3, stride=stride, groups=1),
    'add_k3_e1_g2' : lambda C_in, C_out, layer_id, stride: AddBlock(C_in, C_out, layer_id, expansion=1, kernel_size=3, stride=stride, groups=2),
    'add_k3_e3' : lambda C_in, C_out, layer_id, stride: AddBlock(C_in, C_out, layer_id, expansion=3, kernel_size=3, stride=stride, groups=1),
    'add_k3_e6' : lambda C_in, C_out, layer_id, stride: AddBlock(C_in, C_out, layer_id, expansion=6, kernel_size=3, stride=stride, groups=1),
    'add_k5_e1' : lambda C_in, C_out, layer_id, stride: AddBlock(C_in, C_out, layer_id, expansion=1, kernel_size=5, stride=stride, groups=1),
    'add_k5_e1_g2' : lambda C_in, C_out, layer_id, stride: AddBlock(C_in, C_out, layer_id, expansion=1, kernel_size=5, stride=stride, groups=2),
    'add_k5_e3' : lambda C_in, C_out, layer_id, stride: AddBlock(C_in, C_out, layer_id, expansion=3, kernel_size=5, stride=stride, groups=1),
    'add_k5_e6' : lambda C_in, C_out, layer_id, stride: AddBlock(C_in, C_out, layer_id, expansion=6, kernel_size=5, stride=stride, groups=1),
    'shift_k3_e1' : lambda C_in, C_out, layer_id, stride: ShiftBlock(C_in, C_out, layer_id, expansion=1, kernel_size=3, stride=stride, groups=1),
    'shift_k3_e1_g2' : lambda C_in, C_out, layer_id, stride: ShiftBlock(C_in, C_out, layer_id, expansion=1, kernel_size=3, stride=stride, groups=2),
    'shift_k3_e3' : lambda C_in, C_out, layer_id, stride: ShiftBlock(C_in, C_out, layer_id, expansion=3, kernel_size=3, stride=stride, groups=1),
    'shift_k3_e6' : lambda C_in, C_out, layer_id, stride: ShiftBlock(C_in, C_out, layer_id, expansion=6, kernel_size=3, stride=stride, groups=1),
    'shift_k5_e1' : lambda C_in, C_out, layer_id, stride: ShiftBlock(C_in, C_out, layer_id, expansion=1, kernel_size=5, stride=stride, groups=1),
    'shift_k5_e1_g2' : lambda C_in, C_out, layer_id, stride: ShiftBlock(C_in, C_out, layer_id, expansion=1, kernel_size=5, stride=stride, groups=2),
    'shift_k5_e3' : lambda C_in, C_out, layer_id, stride: ShiftBlock(C_in, C_out, layer_id, expansion=3, kernel_size=5, stride=stride, groups=1),
    'shift_k5_e6' : lambda C_in, C_out, layer_id, stride: ShiftBlock(C_in, C_out, layer_id, expansion=6, kernel_size=5, stride=stride, groups=1),
    'shiftadd_k3_e1' : lambda C_in, C_out, layer_id, stride: ShiftAddBlock(C_in, C_out, layer_id, expansion=1, kernel_size=3, stride=stride, groups=1),
    'shiftadd_k3_e1_g2' : lambda C_in, C_out, layer_id, stride: ShiftAddBlock(C_in, C_out, layer_id, expansion=1, kernel_size=3, stride=stride, groups=2),
    'shiftadd_k3_e3' : lambda C_in, C_out, layer_id, stride: ShiftAddBlock(C_in, C_out, layer_id, expansion=3, kernel_size=3, stride=stride, groups=1),
    'shiftadd_k3_e6' : lambda C_in, C_out, layer_id, stride: ShiftAddBlock(C_in, C_out, layer_id, expansion=6, kernel_size=3, stride=stride, groups=1),
    'shiftadd_k5_e1' : lambda C_in, C_out, layer_id, stride: ShiftAddBlock(C_in, C_out, layer_id, expansion=1, kernel_size=5, stride=stride, groups=1),
    'shiftadd_k5_e1_g2' : lambda C_in, C_out, layer_id, stride: ShiftAddBlock(C_in, C_out, layer_id, expansion=1, kernel_size=5, stride=stride, groups=2),
    'shiftadd_k5_e3' : lambda C_in, C_out, layer_id, stride: ShiftAddBlock(C_in, C_out, layer_id, expansion=3, kernel_size=5, stride=stride, groups=1),
    'shiftadd_k5_e6' : lambda C_in, C_out, layer_id, stride: ShiftAddBlock(C_in, C_out, layer_id, expansion=6, kernel_size=5, stride=stride, groups=1),
}
