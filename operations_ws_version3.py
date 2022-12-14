from pdb import set_trace as bp
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from thop import profile
# from thop.count_hooks import count_convNd

import sys
from torch.nn.modules.utils import _pair
import os.path as osp
from easydict import EasyDict as edict

from config_search import config
from adder import adder
from deepshift import modules
from deepshift import modules_q
from deepshift import utils as utils
from adder.quantize import quantize, quantize_grad, QuantMeasure, calculate_qparams
from torch.distributions.laplace import Laplace
from torch.distributions.normal import Normal


__all__ = ['ConvBlock', 'Skip', 'ConvNorm', 'OPS']

flops_lookup_table = {}
flops_file_name = osp.join(config.hw_platform_path, "flops_lookup_table.npy")
if osp.isfile(flops_file_name):
    flops_lookup_table = np.load(flops_file_name, allow_pickle=True).item()

# TODO:
WEIGHT_BITS = 16

def sub_filter_start_end(kernel_size, sub_kernel_size):
    center = kernel_size // 2
    dev = sub_kernel_size // 2
    start, end = center - dev, center + dev + 1
    assert end - start == sub_kernel_size
    return start, end

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

    def forward(self, input, ratio_out=1, ratio_in=1, ratio_g=1):
        input_q = self.quantize_input_fw(input, WEIGHT_BITS)
        weight_qparams = calculate_qparams(self.weight[:(self.output_channels//ratio_out),:(self.input_channels//ratio_in),:,:], num_bits=WEIGHT_BITS, flatten_dims=(1, -1), reduce_dim=None)
        self.qweight = quantize(self.weight[:(self.output_channels//ratio_out),:(self.input_channels//ratio_in),:,:], qparams=weight_qparams)

        output = F.conv2d(input_q, self.qweight, None, self.stride,
                              self.padding, self.dilation, self.groups//ratio_g)
        
        # TODO:change to inference quantization
        output = quantize_grad(output, num_bits=WEIGHT_BITS, flatten_dims=(1, -1))
        # output = self.quantize_input_fw(output, WEIGHT_BITS)

        # if self.bias is not False : 
        #     output += self.b.unsqueeze(0).unsqueeze(2).unsqueeze(3) 
        #     # output = act_quant(output)

        return output

# TODO:
# Conv2d = nn.Conv2d
# Conv2d = QConv2d
# BatchNorm2d = nn.BatchNorm2d

# FIXME: depthwise conv

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

class Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False,
                 ):

        # kernel_size = _pair(kernel_size)
        # stride = _pair(stride)
        # padding = _pair(padding)
        # dilation = _pair(dilation)
        # self.groups = groups

        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride=stride,
                                      padding=padding, dilation=dilation, groups=groups, bias=bias)
        # self.stride = stride
        # self.padding = padding
        self.input_channels = in_channels
        self.output_channels = out_channels
        self.kernel_size = kernel_size
        self.groups = groups
        
        self.weight = torch.nn.Parameter(
            nn.init.normal_(torch.randn(
                out_channels,in_channels//groups,kernel_size,kernel_size)))

    def forward(self, input, ratio_out=1, ratio_in=1, ratio_g=1, kernel=None):
        sample_weight = self.weight[ :(self.output_channels//ratio_out),:,:,:]
        sample_weight = sample_weight[:, :(self.input_channels//ratio_in),:,:]
        if (kernel!=None):
            start, end = sub_filter_start_end(5, kernel)
            sample_weight = sample_weight[:,:, start:end, start:end]
            padding=(kernel//2,kernel//2)
        else:
            padding=self.padding
        output = F.conv2d(input, sample_weight, None, self.stride, padding, self.dilation, (self.groups//ratio_g))

        return output

class BatchNorm2d(nn.Module):
    def __init__(self, feature_dim):
        super(BatchNorm2d, self).__init__()
        
        self.feature_dim = feature_dim
        self.bn = nn.BatchNorm2d(self.feature_dim)

        #self.exponential_average_factor = 0 #doesn't acculate bn stats
        self.need_sync = False

        # reserved to tracking the performance of the largest and smallest network
        # self.bn_tracking = nn.ModuleList(
        #     [
        #         nn.BatchNorm2d(self.feature_dim, affine=False),
        #         nn.BatchNorm2d(self.feature_dim, affine=False) 
        #     ]
        # )

    def forward(self, x, ratio=1):
        # if not self.training:
        #     raise ValueError('DynamicBN only supports training')
        # return F.batch_norm(x, running_mean=self.running_mean[:(self.super_embed_dim//ratio)], running_var=self.running_var[:(self.super_embed_dim//ratio)], weight=self.weight[:(self.super_embed_dim//ratio)], bias=self.bias[:(self.super_embed_dim//ratio)], eps=self.eps)
        bn = self.bn
        if not self.need_sync:
            return F.batch_norm(
                x, bn.running_mean[:self.feature_dim//ratio], bn.running_var[:self.feature_dim//ratio], bn.weight[:self.feature_dim//ratio],
                bn.bias[:self.feature_dim//ratio], bn.training or not bn.track_running_stats,
                bn.momentum, bn.eps,
            )


class ConvBlock(nn.Module):
    '''
    conv => norm => activation
    use native Conv2d, not slimmable
    '''
    def __init__(self, C_in, C_out,  layer_id, expansion=1, kernel_size=3, stride=1, padding=None, dilation=1, groups=1, bias=False, shared_weight=None):
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
        self.shared_weight = shared_weight

        if self.groups > 1:
            self.shuffle = ChannelShuffle(self.groups)

        self.conv1 = Conv2d(C_in, C_in*expansion, kernel_size=1, stride=1, padding=0, dilation=1, groups=self.groups, bias=bias)
        del self.conv1.weight
        self.bn1 = BatchNorm2d(C_in*expansion)

        self.conv2 = Conv2d(C_in*expansion, C_in*expansion, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=1, groups=C_in*expansion, bias=bias)
        del self.conv2.weight
        self.bn2 = BatchNorm2d(C_in*expansion)
        
        self.conv3 = Conv2d(C_in*expansion, C_out, kernel_size=1, stride=1, padding=0, dilation=1, groups=self.groups, bias=bias)
        del self.conv3.weight
        self.bn3 = BatchNorm2d(C_out)

        self.nl = nn.ReLU(inplace=True)

        self.normal = Normal(0, 1)

    # beta, mode, act_num, beta_param are for bit-widths search
    def forward(self, x, ratio=1, kernel=None):
        # print('ok')
        
        kl_loss = 0
        
        identity = x
        # print(x.size())
        # print('forward')
        # print(self.conv1(x,ratio_out=ratio))
        self.conv1.weight = self.shared_weight.conv1
        x = self.nl(self.bn1(self.conv1(x,ratio_out=ratio),ratio))
        # kl_loss += F.kl_div(F.log_softmax(self.shared_weight.conv1.reshape(-1), 0),
        #                     F.softmax(self.normal.sample(self.shared_weight.conv1.reshape(-1).shape).cuda(), 0), reduction="none").mean()

        if self.groups > 1:
            x = self.shuffle(x)

        self.conv2.weight = self.shared_weight.conv2
        x = self.nl(self.bn2(self.conv2(x,ratio_out=ratio,ratio_g=ratio, kernel=kernel),ratio))
        # print(x.size())
        # TODO:
        # kl_loss += F.kl_div(F.log_softmax(self.shared_weight.conv2.reshape(-1), 0), F.softmax(self.normal.sample(self.shared_weight.conv2.reshape(-1).shape).cuda(), 0), reduction="none").mean()

        self.conv3.weight = self.shared_weight.conv3
        x = self.bn3(self.conv3(x,ratio_in=ratio))
        # print(x.size())
        # kl_loss += F.kl_div(F.log_softmax(self.shared_weight.conv3.reshape(-1), 0),
        #                     F.softmax(self.normal.sample(self.shared_weight.conv3.reshape(-1).shape).cuda(), 0), reduction="none").mean()

        if self.C_in == self.C_out and self.stride == 1:
            x += identity

        return x, 0

    def set_stage(self, stage):
        assert stage == 'update_weight' or stage == 'update_arch'
        self.conv1.set_stage(stage)
        self.conv2.set_stage(stage)
        self.conv3.set_stage(stage)


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

    @staticmethod
    def _flops(h, w, C_in, C_out, expansion=1, kernel_size=3, stride=1, padding=None, dilation=1, groups=1, bias=False):
        layer_id = 1
        layer = ConvBlock(C_in, C_out, layer_id, expansion, kernel_size, stride, padding, dilation, groups, bias)
        flops, params = profile(layer, inputs=(torch.randn(1, C_in, h, w),))
        return flops

    def forward_flops(self, size, ratio):
        c_in, h_in, w_in = size
        expansion = ratio 

        assert c_in == self.C_in, "c_in %d, self.C_in %d"%(c_in, self.C_in)
        c_out = self.C_out

        if self.stride == 1:
            h_out = h_in; w_out = w_in
        else:
            h_out = h_in // 2; w_out = w_in // 2

        name = "ConvBlock_H%d_W%d_Cin%d_Cout%d_exp%d_kernel%d_stride%d_group%d"%(h_in, w_in, c_in, c_out, expansion, self.kernel_size, self.stride, self.groups)
        if name in flops_lookup_table:
            flops = flops_lookup_table[name]
        else:
            print("not found in flops_lookup_table:", name)
            flops = ConvBlock._flops(h_in, w_in, c_in, c_out, expansion, self.kernel_size, self.stride, self.padding, self.dilation, self.groups, self.bias)
            flops_lookup_table[name] = flops
            np.save(flops_file_name, flops_lookup_table)

        return flops, (c_out, h_out, w_out)


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


    def forward(self, x):
        if hasattr(self, 'conv'):
            out = self.conv(x)
            out = self.bn(out)
            out = self.relu(out)
        else:
            out = x

        return out, 0

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

    @staticmethod
    def _flops(h, w, C_in, C_out, stride=1):
        layer = Skip(C_in, C_out, stride)
        flops, params = profile(layer, inputs=(torch.randn(1, C_in, h, w),))
        return flops

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


def Shiftlayer(in_planes, out_planes, kernel_size=1, stride=1, padding=1, groups=1, bias=False, freeze_sign = False, use_kernel=False, use_cuda=True, shift_type='Q',
    rounding='deterministic', weight_bits=6, sign_threshold_ps=None, quant_bits=16):
    # conversion_count = 0
    if shift_type == 'Q':
        shift_conv2d = modules_q.Conv2dShiftQ(in_channels=in_planes, out_channels=out_planes, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=bias, use_kernel=use_kernel, use_cuda=use_cuda, rounding=rounding, weight_bits=weight_bits,quant_bits=quant_bits)
        # shift_conv2d.weight = conv2d.weight
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
    def forward(self, x, ratio=1, kernel=None):
        identity = x
        # print(x.size())
        x = self.nl(self.bn1(self.conv1(x,ratio_out=ratio),ratio))
        # print(self.kernel_size)
        # print(self.padding)
        # print(x.size())

        if self.groups > 1:
            x = self.shuffle(x)

        x = self.nl(self.bn2(self.conv2(x,ratio_out=ratio,ratio_g=ratio,kernel=kernel),ratio))
        # print(x.size())

        x = self.bn3(self.conv3(x,ratio_in=ratio))
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

    def forward_flops(self, size, ratio):
        c_in, h_in, w_in = size
        expansion = ratio

        assert c_in == self.C_in, "c_in %d, self.C_in %d"%(c_in, self.C_in)
        c_out = self.C_out

        if self.stride == 1:
            h_out = h_in; w_out = w_in
        else:
            h_out = h_in // 2; w_out = w_in // 2

        name = "ConvBlock_H%d_W%d_Cin%d_Cout%d_exp%d_kernel%d_stride%d_group%d"%(h_in, w_in, c_in, c_out, expansion, self.kernel_size, self.stride, self.groups)
        if name in flops_lookup_table:
            flops = flops_lookup_table[name]
        else:
            print("not found in flops_lookup_table:", name)
            flops = ConvBlock._flops(h_in, w_in, c_in, c_out, expansion, self.kernel_size, self.stride, self.padding, self.dilation, self.groups, self.bias)
            flops_lookup_table[name] = flops
            np.save(flops_file_name, flops_lookup_table)

        # return flops*0.6, (c_out, h_out, w_out)
        return flops/8, (c_out, h_out, w_out)


def mapping(shared_weight, stage, affine):
    weight_shape = shared_weight.shape
    weight = shared_weight.clone().detach().reshape(-1)
    # weight = shared_weight.reshape(-1)
    weight_min = weight.min()
    weight_max = weight.max()
    for i in range(stage):
        # temp = weight.detach() * self.conv3_affine[i]
        temp = weight * affine[i]
        index = (weight >= (weight_min + i * (weight_max - weight_min) / stage)) & (weight < (weight_min + (i+1) * (weight_max - weight_min) / stage))
        temp =  temp * index.detach()

        if i == 0:
            new_weight = temp
        else:
            new_weight = new_weight + temp
            del temp

        return new_weight.reshape(weight_shape)

class AddBlock(nn.Module):
    '''
    conv => norm => activation
    use native Conv2d, not slimmable
    '''
    def __init__(self, C_in, C_out, layer_id, expansion=1, kernel_size=3, stride=1, padding=None, dilation=1, groups=1, bias=False, shared_weight=None):
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
        self.shared_weight = shared_weight

        if self.groups > 1:
            self.shuffle = ChannelShuffle(self.groups)

        self.conv1 = Addlayer(C_in, C_in*expansion, kernel_size=1, stride=1, padding=0, groups=self.groups)
        del self.conv1.adder
        self.bn1 = BatchNorm2d(C_in*expansion)

        self.conv2 = Addlayer(C_in*expansion, C_in*expansion, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, groups=C_in*expansion)
        self.bn2 = BatchNorm2d(C_in*expansion)
        del self.conv2.adder

        self.conv3 = Addlayer(C_in*expansion, C_out, kernel_size=1, stride=1, padding=0, groups=self.groups)
        self.bn3 = BatchNorm2d(C_out)
        del self.conv3.adder

        self.nl = nn.ReLU(inplace=True)

        # piece-wise affine
        self.stage = 100
        self.conv1_affine = nn.Parameter(torch.randn(self.stage))
        self.conv2_affine = nn.Parameter(torch.randn(self.stage))
        self.conv3_affine = nn.Parameter(torch.randn(self.stage))

        # TODO: 
        self.laplace = Laplace(0, 2)

    # beta, mode, act_num, beta_param are for bit-widths search
    def forward(self, x, ratio=1, kernel=None):
        identity = x
        # print(x.size())
        kl_loss = 0

        self.conv1.adder = mapping(self.shared_weight.conv1, self.stage, self.conv1_affine)
        x = self.nl(self.bn1(self.conv1(x,ratio_out=ratio),ratio))
        kl_loss += F.kl_div(F.log_softmax(self.conv1.adder, 0), F.softmax(self.laplace.sample(self.conv1.adder.shape).cuda(), 0), reduction="none").mean()

        if self.groups > 1:
            x = self.shuffle(x)
        
        self.conv2.adder = mapping(self.shared_weight.conv2, self.stage, self.conv2_affine)
        x = self.nl(self.bn2(self.conv2(x,ratio_out=ratio,ratio_g=ratio,kernel=kernel),ratio))
        # TODO:
        kl_loss += F.kl_div(F.log_softmax(self.conv2.adder, 0), F.softmax(self.laplace.sample(self.conv2.adder.shape).cuda(), 0), reduction="none").mean()

        self.conv3.adder = mapping(self.shared_weight.conv3, self.stage, self.conv3_affine)
        x = self.bn3(self.conv3(x,ratio_in=ratio))
        kl_loss += F.kl_div(F.log_softmax(self.conv3.adder, 0), F.softmax(self.laplace.sample(self.conv3.adder.shape).cuda(), 0), reduction="none").mean()

        # print(x.size())
        if self.C_in == self.C_out and self.stride == 1:
            x += identity

        return x, kl_loss

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

    def forward_flops(self, size, ratio):
        c_in, h_in, w_in = size
        expansion = ratio

        assert c_in == self.C_in, "c_in %d, self.C_in %d"%(c_in, self.C_in)
        c_out = self.C_out

        if self.stride == 1:
            h_out = h_in; w_out = w_in
        else:
            h_out = h_in // 2; w_out = w_in // 2

        name = "ConvBlock_H%d_W%d_Cin%d_Cout%d_exp%d_kernel%d_stride%d_group%d"%(h_in, w_in, c_in, c_out, expansion, self.kernel_size, self.stride, self.groups)
        if name in flops_lookup_table:
            flops = flops_lookup_table[name]
        else:
            print("not found in flops_lookup_table:", name)
            flops = ConvBlock._flops(h_in, w_in, c_in, c_out, expansion, self.kernel_size, self.stride, self.padding, self.dilation, self.groups, self.bias)
            flops_lookup_table[name] = flops
            np.save(flops_file_name, flops_lookup_table)

        # return flops*0.6, (c_out, h_out, w_out)
        return flops/5, (c_out, h_out, w_out)

class AddBlock_nodpws(nn.Module):
    '''
    conv => norm => activation
    use native Conv2d, not slimmable
    '''
    def __init__(self, C_in, C_out, layer_id, expansion=1, kernel_size=3, stride=1, padding=None, dilation=1, groups=1, bias=False, shared_weight=None):
        super(AddBlock_nodpws, self).__init__()
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
        self.shared_weight = shared_weight

        if self.groups > 1:
            self.shuffle = ChannelShuffle(self.groups)

        self.conv1 = Addlayer(C_in, C_in*expansion, kernel_size=1, stride=1, padding=0, groups=self.groups)
        del self.conv1.adder
        self.bn1 = BatchNorm2d(C_in*expansion)

        # self.conv2 = Addlayer(C_in*expansion, C_in*expansion, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, groups=C_in*expansion)
        self.conv2 = Conv2d(C_in*expansion, C_in*expansion, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=1, groups=C_in*expansion, bias=bias)
        self.bn2 = BatchNorm2d(C_in*expansion)
        del self.conv2.weight

        self.conv3 = Addlayer(C_in*expansion, C_out, kernel_size=1, stride=1, padding=0, groups=self.groups)
        self.bn3 = BatchNorm2d(C_out)
        del self.conv3.adder

        self.nl = nn.ReLU(inplace=True)

        # piece-wise affine
        self.stage = 100
        self.conv1_affine = nn.Parameter(torch.randn(self.stage))
        # self.conv2_affine = nn.Parameter(torch.randn(self.stage))
        self.conv3_affine = nn.Parameter(torch.randn(self.stage))

        # TODO: 
        self.laplace = Laplace(0, 2)
        self.normal = Normal(0, 1)

    # beta, mode, act_num, beta_param are for bit-widths search
    def forward(self, x, ratio=1, kernel=None):
        identity = x
        # print(x.size())
        kl_loss = 0

        self.conv1.adder = mapping(self.shared_weight.conv1, self.stage, self.conv1_affine)
        x = self.nl(self.bn1(self.conv1(x,ratio_out=ratio),ratio))
        kl_loss += F.kl_div(F.log_softmax(self.conv1.adder, 0), F.softmax(self.laplace.sample(self.conv1.adder.shape).cuda(), 0), reduction="none").mean()

        if self.groups > 1:
            x = self.shuffle(x)
        
        # self.conv2.adder = mapping(self.shared_weight.conv2, self.stage, self.conv2_affine)
        self.conv2.weight = self.shared_weight.conv2
        x = self.nl(self.bn2(self.conv2(x,ratio_out=ratio,ratio_g=ratio,kernel=kernel),ratio))
        # TODO:
        kl_loss += F.kl_div(F.log_softmax(self.shared_weight.conv2.reshape(-1), 0), F.softmax(self.normal.sample(self.shared_weight.conv2.reshape(-1).shape).cuda(), 0), reduction="none").mean()

        self.conv3.adder = mapping(self.shared_weight.conv3, self.stage, self.conv3_affine)
        x = self.bn3(self.conv3(x,ratio_in=ratio))
        kl_loss += F.kl_div(F.log_softmax(self.conv3.adder, 0), F.softmax(self.laplace.sample(self.conv3.adder.shape).cuda(), 0), reduction="none").mean()

        # print(x.size())
        if self.C_in == self.C_out and self.stride == 1:
            x += identity

        return x, kl_loss

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

    def forward_flops(self, size, ratio):
        c_in, h_in, w_in = size
        expansion = ratio

        assert c_in == self.C_in, "c_in %d, self.C_in %d"%(c_in, self.C_in)
        c_out = self.C_out

        if self.stride == 1:
            h_out = h_in; w_out = w_in
        else:
            h_out = h_in // 2; w_out = w_in // 2

        name = "ConvBlock_H%d_W%d_Cin%d_Cout%d_exp%d_kernel%d_stride%d_group%d"%(h_in, w_in, c_in, c_out, expansion, self.kernel_size, self.stride, self.groups)
        if name in flops_lookup_table:
            flops = flops_lookup_table[name]
        else:
            print("not found in flops_lookup_table:", name)
            flops = ConvBlock._flops(h_in, w_in, c_in, c_out, expansion, self.kernel_size, self.stride, self.padding, self.dilation, self.groups, self.bias)
            flops_lookup_table[name] = flops
            np.save(flops_file_name, flops_lookup_table)

        # return flops*0.6, (c_out, h_out, w_out)
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
    def forward(self, x, ratio=1, kernel=None):
        identity = x
        # print(x.size())
        # x = self.nl(self.bn1(self.conv1(x)))
        x = self.conv1[0](x, ratio_out=ratio)
        x = self.conv1[1](x, ratio_out=ratio, ratio_in=ratio)
        x = self.nl(self.bn1(x,ratio))


        if self.groups > 1:
            x = self.shuffle(x)

        # x = self.nl(self.bn2(self.conv2(x)))
        x = self.conv2[0](x, ratio_out=ratio, ratio_g=ratio, kernel=kernel)
        x = self.conv2[1](x, ratio_out=ratio, ratio_g=ratio, kernel=kernel)
        x = self.nl(self.bn2(x,ratio))
        # print(x.size())

        # x = self.bn3(self.conv3(x))
        x = self.conv3[0](x, ratio_in=ratio)
        # x = self.conv3.modules[1](x)
        x = self.bn3(self.conv3[1](x))
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

    def forward_flops(self, size, ratio):
        c_in, h_in, w_in = size
        expansion = ratio
        assert c_in == self.C_in, "c_in %d, self.C_in %d"%(c_in, self.C_in)
        c_out = self.C_out

        if self.stride == 1:
            h_out = h_in; w_out = w_in
        else:
            h_out = h_in // 2; w_out = w_in // 2

        name = "ShiftAddBlock_H%d_W%d_Cin%d_Cout%d_exp%d_kernel%d_stride%d_group%d"%(h_in, w_in, c_in, c_out, expansion, self.kernel_size, self.stride, self.groups)
        if name in flops_lookup_table:
            flops = flops_lookup_table[name]
        else:
            print("not found in flops_lookup_table:", name)
            
            # ############### shift layer flops #########
            name_shift = "ConvBlock_H%d_W%d_Cin%d_Cout%d_exp%d_kernel%d_stride%d_group%d"%(h_in, w_in, c_in, c_out, expansion, self.kernel_size, self.stride, self.groups)
            if name_shift in flops_lookup_table:
                flops_shift = flops_lookup_table[name_shift]
            else:
                flops_shift = ConvBlock._flops(h_in, w_in, c_in, c_out, expansion, self.kernel_size, self.stride, self.padding, self.dilation, self.groups, self.bias)
                flops_lookup_table[name_shift] = flops_shift
            # np.save(flops_file_name, flops_lookup_table)

            # ############### add layer flops #########
            layer = Addlayer(c_in*expansion, c_in*expansion, kernel_size=1, stride=1, padding=0)
            flop, params = profile(layer, inputs=(torch.randn(1, c_in*expansion, h_in, w_in),))
            flops_add = flop
            layer = Addlayer(c_in*expansion, c_in*expansion, kernel_size=self.kernel_size, stride=1, padding=self.padding)
            flop, params = profile(layer, inputs=(torch.randn(1, c_in*expansion, h_out, w_out),))
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

        self.conv = nn.Conv2d(C_in, C_out, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, 
                            dilation=self.dilation, groups=self.groups, bias=bias)
        self.bn = nn.BatchNorm2d(C_out)

        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        return x


    def set_stage(self, stage):
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

        conv_info = [
                    [self.stride, {'ch_out':[self.C_out,0],'ch_in':[self.C_in,0],'batch':[1,0],'col_out':[h_out,0],
                    'row_out':[w_out,0],'row_kernel':[self.kernel_size, 0],'col_kernel':[self.kernel_size,0]}, 0, 1]
                    ]
            
        return conv_info, (self.C_out, h_out, w_out)

    @staticmethod
    def _flops(h, w, C_in, C_out, kernel_size=3, stride=1, padding=None, dilation=1, groups=1, bias=False):
        layer = ConvNorm(C_in, C_out, kernel_size, stride, padding, dilation, groups, bias)
        flops, params = profile(layer, inputs=(torch.randn(1, C_in, h, w),))
        return flops

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



OPS = {
    'k3_e1' : lambda C_in, C_out, layer_id, stride, shared_parameters: ConvBlock(C_in, C_out, layer_id, expansion=1, kernel_size=3, stride=stride, groups=1, shared_weight=shared_parameters),
    'k3_e1_g2' : lambda C_in, C_out, layer_id, stride, shared_parameters: ConvBlock(C_in, C_out, layer_id, expansion=1, kernel_size=3, stride=stride, groups=2, shared_weight=shared_parameters),
    'k3_e3' : lambda C_in, C_out, layer_id, stride, shared_parameters: ConvBlock(C_in, C_out, layer_id, expansion=3, kernel_size=3, stride=stride, groups=1, shared_weight=shared_parameters),
    'k3_e6' : lambda C_in, C_out, layer_id, stride, shared_parameters: ConvBlock(C_in, C_out, layer_id, expansion=6, kernel_size=3, stride=stride, groups=1, shared_weight=shared_parameters),
    'k5_e1' : lambda C_in, C_out, layer_id, stride, shared_parameters: ConvBlock(C_in, C_out, layer_id, expansion=1, kernel_size=5, stride=stride, groups=1, shared_weight=shared_parameters),
    'k5_e1_g2' : lambda C_in, C_out, layer_id, stride, shared_parameters: ConvBlock(C_in, C_out, layer_id, expansion=1, kernel_size=5, stride=stride, groups=2, shared_weight=shared_parameters),
    'k5_e3' : lambda C_in, C_out, layer_id, stride, shared_parameters: ConvBlock(C_in, C_out, layer_id, expansion=3, kernel_size=5, stride=stride, groups=1, shared_weight=shared_parameters),
    'k5_e6' : lambda C_in, C_out, layer_id, stride, shared_parameters: ConvBlock(C_in, C_out, layer_id, expansion=6, kernel_size=5, stride=stride, groups=1, shared_weight=shared_parameters),
    'skip' : lambda C_in, C_out, layer_id, stride, shared_parameters: Skip(C_in, C_out, layer_id, stride),
    'add_k3_e1' : lambda C_in, C_out, layer_id, stride, shared_parameters: AddBlock(C_in, C_out, layer_id, expansion=1, kernel_size=3, stride=stride, groups=1, shared_weight=shared_parameters),
    'add_k3_e1_g2' : lambda C_in, C_out, layer_id, stride, shared_parameters: AddBlock(C_in, C_out, layer_id, expansion=1, kernel_size=3, stride=stride, groups=2, shared_weight=shared_parameters),
    'add_k3_e3' : lambda C_in, C_out, layer_id, stride, shared_parameters: AddBlock(C_in, C_out, layer_id, expansion=3, kernel_size=3, stride=stride, groups=1, shared_weight=shared_parameters),
    'add_k3_e6' : lambda C_in, C_out, layer_id, stride, shared_parameters: AddBlock(C_in, C_out, layer_id, expansion=6, kernel_size=3, stride=stride, groups=1, shared_weight=shared_parameters),
    'add_k5_e1' : lambda C_in, C_out, layer_id, stride, shared_parameters: AddBlock(C_in, C_out, layer_id, expansion=1, kernel_size=5, stride=stride, groups=1, shared_weight=shared_parameters),
    'add_k5_e1_g2' : lambda C_in, C_out, layer_id, stride, shared_parameters: AddBlock(C_in, C_out, layer_id, expansion=1, kernel_size=5, stride=stride, groups=2, shared_weight=shared_parameters),
    'add_k5_e3' : lambda C_in, C_out, layer_id, stride, shared_parameters: AddBlock(C_in, C_out, layer_id, expansion=3, kernel_size=5, stride=stride, groups=1, shared_weight=shared_parameters),
    'add_k5_e6' : lambda C_in, C_out, layer_id, stride, shared_parameters: AddBlock(C_in, C_out, layer_id, expansion=6, kernel_size=5, stride=stride, groups=1, shared_weight=shared_parameters),
    'shift_k3_e1' : lambda C_in, C_out, layer_id, stride, shared_parameters: ShiftBlock(C_in, C_out, layer_id, expansion=1, kernel_size=3, stride=stride, groups=1, shared_weight=shared_parameters),
    'shift_k3_e1_g2' : lambda C_in, C_out, layer_id, stride, shared_parameters: ShiftBlock(C_in, C_out, layer_id, expansion=1, kernel_size=3, stride=stride, groups=2, shared_weight=shared_parameters),
    'shift_k3_e3' : lambda C_in, C_out, layer_id, stride, shared_parameters: ShiftBlock(C_in, C_out, layer_id, expansion=3, kernel_size=3, stride=stride, groups=1, shared_weight=shared_parameters),
    'shift_k3_e6' : lambda C_in, C_out, layer_id, stride, shared_parameters: ShiftBlock(C_in, C_out, layer_id, expansion=6, kernel_size=3, stride=stride, groups=1, shared_weight=shared_parameters),
    'shift_k5_e1' : lambda C_in, C_out, layer_id, stride, shared_parameters: ShiftBlock(C_in, C_out, layer_id, expansion=1, kernel_size=5, stride=stride, groups=1, shared_weight=shared_parameters),
    'shift_k5_e1_g2' : lambda C_in, C_out, layer_id, stride, shared_parameters: ShiftBlock(C_in, C_out, layer_id, expansion=1, kernel_size=5, stride=stride, groups=2, shared_weight=shared_parameters),
    'shift_k5_e3' : lambda C_in, C_out, layer_id, stride, shared_parameters: ShiftBlock(C_in, C_out, layer_id, expansion=3, kernel_size=5, stride=stride, groups=1, shared_weight=shared_parameters),
    'shift_k5_e6' : lambda C_in, C_out, layer_id, stride, shared_parameters: ShiftBlock(C_in, C_out, layer_id, expansion=6, kernel_size=5, stride=stride, groups=1, shared_weight=shared_parameters),
    'shiftadd_k3_e1' : lambda C_in, C_out, layer_id, stride, shared_parameters: ShiftAddBlock(C_in, C_out, layer_id, expansion=1, kernel_size=3, stride=stride, groups=1, shared_weight=shared_parameters),
    'shiftadd_k3_e1_g2' : lambda C_in, C_out, layer_id, stride, shared_parameters: ShiftAddBlock(C_in, C_out, layer_id, expansion=1, kernel_size=3, stride=stride, groups=2, shared_weight=shared_parameters),
    'shiftadd_k3_e3' : lambda C_in, C_out, layer_id, stride, shared_parameters: ShiftAddBlock(C_in, C_out, layer_id, expansion=3, kernel_size=3, stride=stride, groups=1, shared_weight=shared_parameters),
    'shiftadd_k3_e6' : lambda C_in, C_out, layer_id, stride, shared_parameters: ShiftAddBlock(C_in, C_out, layer_id, expansion=6, kernel_size=3, stride=stride, groups=1, shared_weight=shared_parameters),
    'shiftadd_k5_e1' : lambda C_in, C_out, layer_id, stride, shared_parameters: ShiftAddBlock(C_in, C_out, layer_id, expansion=1, kernel_size=5, stride=stride, groups=1, shared_weight=shared_parameters),
    'shiftadd_k5_e1_g2' : lambda C_in, C_out, layer_id, stride, shared_parameters: ShiftAddBlock(C_in, C_out, layer_id, expansion=1, kernel_size=5, stride=stride, groups=2, shared_weight=shared_parameters),
    'shiftadd_k5_e3' : lambda C_in, C_out, layer_id, stride, shared_parameters: ShiftAddBlock(C_in, C_out, layer_id, expansion=3, kernel_size=5, stride=stride, groups=1, shared_weight=shared_parameters),
    'shiftadd_k5_e6' : lambda C_in, C_out, layer_id, stride, shared_parameters: ShiftAddBlock(C_in, C_out, layer_id, expansion=6, kernel_size=5, stride=stride, groups=1, shared_weight=shared_parameters),
}
