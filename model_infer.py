import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES_OnlyConv, PRIMITIVES_AddAdd, PRIMITIVES_AddShift, PRIMITIVES_AddShiftAdd, PRIMITIVES_AddAll, PRIMITIVES_NoConv, PRIMITIVES_OnlyShiftAdd
import numpy as np
from thop import profile
from matplotlib import pyplot as plt
from thop import profile

#from fpga_nips import search_opt_hw_rs as search_opt_hw
# from fpga_nips import search_opt_hw_diff as search_opt_hw

#from analytical_model.analytical_prediction import search_for_best_latency, evaluate_latency


class MixedOp(nn.Module):
    def __init__(self, C_in, C_out, op_idx, layer_id, stride=1, search_space='OnlyConv', flag=False):
        super(MixedOp, self).__init__()
        self.layer_id = layer_id
        self.search_space = search_space
        if(self.search_space=='OnlyConv'):
            self.type = PRIMITIVES_OnlyConv
        if(self.search_space=='AddAdd'):
            self.type = PRIMITIVES_AddAdd
        if(self.search_space=='AddShift'):
            self.type = PRIMITIVES_AddShift
        if(self.search_space=='AddShiftAdd'):
            self.type = PRIMITIVES_AddShiftAdd
        if(self.search_space=='AddAll'):
            self.type = PRIMITIVES_AddAll
        if(self.search_space=='NoConv'):
            self.type = PRIMITIVES_NoConv
        if(self.search_space=='OnlyShiftAdd'):
            self.type = PRIMITIVES_OnlyShiftAdd
        
        if flag == True:
            if(op_idx==(len(self.type)-1)):
                self._op = OPS[self.type[op_idx]](C_in, C_out, layer_id, stride)
            else:
                op_idx = op_idx % 6 
                self._op = OPS[self.type[op_idx]](C_in, C_out, layer_id, stride)
        else:
            self._op = OPS[self.type[op_idx]](C_in, C_out, layer_id, stride)
        print(self.type[op_idx])
        # print(op_idx)
        self.num_bits = 32

    #    if type(num_bits_list) == list:
    #        self.num_bits = num_bits_list[quant_idx[op_idx]]
    #    else:
    #        self.num_bits = num_bits_list

    def forward(self, x):
        return self._op(x)
        # if(self.layer_id == 1):
        #     print(self.type[op_idx])
        #     print("input:":)

    def forward_energy(self, size):
        # int: force #channel; tensor: arch_ratio; float(<=1): force width
        energy, size_out = self._op.forward_energy(size)
        return energy, size_out

    def forward_latency(self, size):
        # int: force #channel; tensor: arch_ratio; float(<=1): force width
        latency, size_out = self._op.forward_latency(size)
        return latency, size_out

    def forward_flops(self, size):
        # int: force #channel; tensor: arch_ratio; float(<=1): force width
        flops, size_out = self._op.forward_flops(size)
        
        return flops, size_out

    def forward_bitops(self, size):
        flops, size_out = self._op.forward_flops(size)

        bitops = flops * self.num_bits * self.num_bits

        return bitops, size_out

    def layer_info(self, size):
        block_info, size = self._op.layer_info(size)

        return block_info, size


class FBNet_Infer(nn.Module):
    def __init__(self, alpha, config, flag=False, cand=None):
        super(FBNet_Infer, self).__init__()

        if (cand == None):
            self.op_idx_list = F.softmax(alpha, dim=-1).argmax(-1)
            # self.quant_idx_list = F.softmax(beta, dim=-1).argmax(-1)
        else:
            self.op_idx_list = cand

        self.num_classes = config.num_classes

        self.num_layer_list = config.num_layer_list
        self.num_channel_list = config.num_channel_list
        self.stride_list = config.stride_list

#        self.num_bits_list = config.num_bits_list

        self.stem_channel = config.stem_channel
        self.header_channel = config.header_channel
        self.search_space = config.search_space

        if config.dataset == 'imagenet':
            stride_init = 2
        else:
            stride_init = 1

        self.stem = ConvNorm(3, self.stem_channel, kernel_size=3, stride=stride_init, padding=1, bias=False)

        self.cells = nn.ModuleList()

        layer_id = 1

        for stage_id, num_layer in enumerate(self.num_layer_list):
            for i in range(num_layer):
                if i == 0:
                    if stage_id == 0:
                        op = MixedOp(self.stem_channel, self.num_channel_list[stage_id], self.op_idx_list[layer_id-1], layer_id, stride=self.stride_list[stage_id], search_space=self.search_space, flag=flag)
                    else:
                        op = MixedOp(self.num_channel_list[stage_id-1], self.num_channel_list[stage_id], self.op_idx_list[layer_id-1], layer_id, stride=self.stride_list[stage_id], search_space=self.search_space, flag=flag)
                else:
                    op = MixedOp(self.num_channel_list[stage_id], self.num_channel_list[stage_id], self.op_idx_list[layer_id-1], layer_id, stride=1, search_space=self.search_space, flag=flag)
                
                layer_id += 1
                self.cells.append(op)

        self.header = ConvNorm(self.num_channel_list[-1], self.header_channel, kernel_size=1)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.header_channel, self.num_classes)

        self._criterion = nn.CrossEntropyLoss()

        self.init_params()


    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


    def forward(self, input):

        out = self.stem(input)
        middle_out = []

        for i, cell in enumerate(self.cells):
            out = cell(out)
            if i == 4 or i == 8 or i == 12 or i == 16 or i == 20:
                middle_out.append(out)

        out = self.fc(self.avgpool(self.header(out)).view(out.size(0), -1))

        return out, middle_out
    
    def forward_flops(self, size):

        flops_total = []

        flops, size = self.stem.forward_flops(size)
        flops_total.append(flops)

        for i, cell in enumerate(self.cells):
            flops, size = cell.forward_flops(size)
            flops_total.append(flops)

        flops, size = self.header.forward_flops(size)
        flops_total.append(flops)

        return sum(flops_total)


    def forward_bitops(self, size):

        bitops_total = []

        flops, size = self.stem.forward_flops(size)
        bitops_total.append(flops*8*8)

        for i, cell in enumerate(self.cells):
            bitops, size = cell.forward_bitops(size)
            bitops_total.append(bitops)

        flops, size = self.header.forward_flops(size)
        bitops_total.append(flops*8*8)

        return sum(bitops_total)


#    def forward_edp(self, size):
#        energy = self.forward_energy(size)
#        latency = self.forward_latency(size)
#
#        return energy * latency


    def forward_energy(self, size):

        energy_total = []

        energy, size = self.stem.forward_energy(size)
        energy_total.append(energy)

        for i, cell in enumerate(self.cells):
            energy, size = cell.forward_energy(size)
            energy_total.append(energy)

        energy, size = self.header.forward_energy(size)
        energy_total.append(energy)

        return sum(energy_total)


    def forward_latency(self, size):
        
        latency_total = []

        latency, size = self.stem.forward_latency(size)
        latency_total.append(latency)

        for i, cell in enumerate(self.cells):
            latency, size = cell.forward_latency(size)
            latency_total.append(latency)

        latency, size = self.header.forward_latency(size)
        latency_total.append(latency)

        return sum(latency_total)


    def _loss(self, input, target):
        logit = self(input)
        loss = self._criterion(logit, target)

        return loss


    def layer_info(self, size):
        arch_info = []

        block_info, size = self.stem.layer_info(size)
        arch_info.extend(block_info)

        for i, cell in enumerate(self.cells):
            block_info, size = cell.layer_info(size)
            arch_info.extend(block_info)

        block_info, size = self.header.layer_info(size)
        arch_info.extend(block_info)

        return arch_info


