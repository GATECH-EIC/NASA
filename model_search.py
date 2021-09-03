import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.nn import init
from operations import *
import operations
from torch.autograd import Variable
from genotypes import PRIMITIVES_OnlyConv, PRIMITIVES_AddAdd, PRIMITIVES_AddShift, PRIMITIVES_AddShiftAdd, PRIMITIVES_AddAll, PRIMITIVES_NoConv, PRIMITIVES_OnlyShiftAdd
import numpy as np
from thop import profile
from matplotlib import pyplot as plt
from thop import profile

def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    U = U.cuda()
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature=1):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature=1, hard=False):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(F.log_softmax(logits, dim=-1), temperature)

    return y
    
def freeze(layer):
    for child in layer.children():
        for param in child.parameters():
            param.requires_grad = False


def Activate(layer):
    for child in layer.children():
        for param in child.parameters():
            param.requires_grad = True


class MixedOp(nn.Module):

    def __init__(self, C_in, C_out, layer_id, stride=1, mode='soft', act_num=1, search_space='OnlyConv'):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        self.layer_id = layer_id
        self.mode = mode
        self.act_num = act_num
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

        for primitive in self.type:
            op = OPS[primitive](C_in, C_out, layer_id, stride)
            self._ops.append(op)

        self.register_buffer('active_list', torch.tensor(list(range(len(self._ops)))))


    def forward(self, x, alpha, alpha_param=None, update_arch=True, full_channel=False, cand=None):
        # int: force #channel; tensor: arch_ratio; float(<=1): force width
        result = 0

        
        if self.mode == 'soft':
            for i, (w, op) in enumerate(zip(alpha, self._ops)):
                result = result + op(x) * w 

            self.set_active_list(list(range(len(self._ops))))

        elif self.mode == 'proxy_hard':
            if (cand==None):
                assert alpha_param is not None
                rank = alpha.argsort(descending=True)
                if (update_arch == False):
                #     random.shuffle(rank)
                    np.random.shuffle(rank.cpu().detach().numpy())
                self.set_active_list(rank[:self.act_num])

                alpha = F.softmax(alpha_param[rank[:self.act_num]], dim=-1)
                
                # print("self.act_num, len(rank)",self.act_num, len(rank))
                
                # TODO: change ((1-alpha[0]).detach() + alpha[0]) to alpha[0]
                result = result + self._ops[rank[0]](x) * alpha[0]
                # result = result + self._ops[rank[0]](x) * ((1-alpha[0]).detach() + alpha[0])
            
                # TODO: change ((0-alpha[i]).detach() + alpha[i]) into alpha[i] to avoid nan. I find that the coefficient of 0 leads to NaN when search sapce include addlayers.
                for i in range(1,self.act_num):
                    result = result + self._ops[rank[i]](x) * alpha[i]
                    # result = result + self._ops[rank[i]](x) * ((0-alpha[i]).detach() + alpha[i])
            else:
                self.set_active_list(cand)
                result = result + self._ops[cand](x) 

        else:
            print('Wrong search mode:', self.mode)
            sys.exit()
        # else:
        #     rank = alpha.argsort(descending=True)
        #     
        #     result = result + self._ops[rank[0]](x) * alpha[0]
        #     for i in range(1,self.act_num):
        #         result = result + self._ops[rank[i]](x) * alpha[i]
        
        # print(result)
        return result

        
    # set the active operator list for each block
    def set_active_list(self, active_list):
        if type(active_list) is not torch.Tensor:
            active_list = torch.tensor(active_list).cuda()

        self.active_list.data = active_list.data


    def set_stage(self, stage):
        assert stage == 'update_weight' or stage == 'update_arch'

        for op in self._ops:
            op.set_stage(stage)


    def forward_energy(self, size, alpha):
        # int: force #channel; tensor: arch_ratio; float(<=1): force width
        result = 0

        for w, op in zip(alpha, self._ops):
            energy, size_out = op.forward_energy(size)
            result = result + energy * w
        return result, size_out


    def forward_latency(self, size, alpha):
        # int: force #channel; tensor: arch_ratio; float(<=1): force width
        result = 0
        
        # TODO：
        for w, op in zip(alpha, self._ops):
            latency, size_out = op.forward_latency(size)
            result = result + latency * w
        return result, size_out


    def forward_flops(self, size, alpha):
        # int: force #channel; tensor: arch_ratio; float(<=1): force width
        result = 0


        op_id = alpha.argsort(descending=True)[0]
        # print(alpha)
        print("op_id:",op_id)
        flops, size_out = self._ops[op_id].forward_flops(size)
        # print("flops",flops)
        # print(alpha[op_id])
        result = alpha[op_id] * flops

        return result, size_out


class FBNet(nn.Module):
    def __init__(self, config):
        super(FBNet, self).__init__()

        self.hard = config.hard

        self.mode = config.mode
        self.act_num = config.act_num

        self.num_classes = config.num_classes

        self.num_layer_list = config.num_layer_list
        self.num_channel_list = config.num_channel_list
        self.stride_list = config.stride_list

        self.stem_channel = config.stem_channel
        self.header_channel = config.header_channel
        self.search_space = config.search_space
        
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
                        op = MixedOp(self.stem_channel, self.num_channel_list[stage_id], layer_id, stride=self.stride_list[stage_id], mode=self.mode, act_num=self.act_num, search_space=self.search_space)
                    else:
                        op = MixedOp(self.num_channel_list[stage_id-1], self.num_channel_list[stage_id], layer_id, stride=self.stride_list[stage_id], mode=self.mode, act_num=self.act_num, search_space=self.search_space)
                else:
                    op = MixedOp(self.num_channel_list[stage_id], self.num_channel_list[stage_id], layer_id, stride=1, mode=self.mode, act_num=self.act_num, search_space=self.search_space)
                
                layer_id += 1
                self.cells.append(op)

        self.header = ConvNorm(self.num_channel_list[-1], self.header_channel, kernel_size=1)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.header_channel, self.num_classes)

        self._arch_params = self._build_arch_parameters()
        self._reset_arch_parameters()

        self._criterion = nn.CrossEntropyLoss()

        self.sample_func = config.sample_func

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


    def forward(self, input, temp=1, update_arch=True, full_channel=False, full_kernel=False, cand=None):
        if self.sample_func == 'softmax':
            alpha = F.softmax(getattr(self, "alpha"), dim=-1)
        else:
            alpha = gumbel_softmax(getattr(self, "alpha"), temperature=temp, hard=self.hard)
    
        out = self.stem(input)

        if (cand==None):
            for i, cell in enumerate(self.cells):
                out = cell(out, alpha[i], getattr(self, "alpha")[i], update_arch, full_channel)
        else:
            for i, cell in enumerate(self.cells):
                out = cell(out, alpha[i], getattr(self, "alpha")[i], update_arch, full_channel, cand[i])
        # print(out)

        
        # TODO:
        out = self.fc(self.avgpool(self.header(out)).view(out.size(0), -1))
        # out = self.header(out)
        # print(out)
        # out = self.avgpool(out)
        # print("out_one",out.viSew(out.size(0), -1))
        # out = self.fc(out.view(out.size(0), -1))
        # print("out_two",out)
        return out
        ###################################


    def set_search_mode(self, mode='soft', act_num=1):
        self.mode = mode
        self.act_num = act_num

        if self.mode == 'soft':
            self.hard = False
        else:
            self.hard = True

        for cell in self.cells:
            cell.mode = mode
            cell.act_num = act_num


    def set_stage(self, stage):
        assert stage == 'update_weight' or stage == 'update_arch'
        for cell in self.cells:
            cell.set_stage(stage)

    def show_arch(self, alpha=None, cands=None):
        # if self.sample_func == 'softmax':
        #     alpha = F.softmax(getattr(self, "alpha"), dim=-1)
        # else:
        #     alpha = gumbel_softmax(getattr(self, "alpha"), temperature=temp, hard=self.hard)
        if alpha != None:
            op_idx_list = F.softmax(alpha, dim=-1).argmax(-1)
        else:
            op_idx_list = cands

        for i, _ in enumerate(self.cells):
            print(self.type[op_idx_list[i]], end=' ')


    def forward_flops(self, size, temp=1):
        if self.sample_func == 'softmax':
            alpha = F.softmax(getattr(self, "alpha"), dim=-1)
        else:
            alpha = gumbel_softmax(getattr(self, "alpha"), temperature=temp, hard=self.hard)

        flops_total = []

        flops, size = self.stem.forward_flops(size)
        flops_total.append(flops)

        for i, cell in enumerate(self.cells):
            flops, size = cell.forward_flops(size, alpha[i])
            flops_total.append(flops)

        flops, size = self.header.forward_flops(size)
        flops_total.append(flops)

        return sum(flops_total)


    def forward_energy(self, size, temp=1):
        if self.sample_func == 'softmax':
            alpha = F.softmax(getattr(self, "alpha"), dim=-1)
        else:
            alpha = gumbel_softmax(getattr(self, "alpha"), temperature=temp, hard=self.hard)

        energy_total = []

        energy, size = self.stem.forward_energy(size)
        energy_total.append(energy)

        for i, cell in enumerate(self.cells):
            energy, size = cell.forward_energy(size, alpha[i])
            energy_total.append(energy)

        energy, size = self.header.forward_energy(size)
        energy_total.append(energy)

        return sum(energy_total)


    def forward_latency(self, size, temp=1):
        if self.sample_func == 'softmax':
            alpha = F.softmax(getattr(self, "alpha"), dim=-1)
        else:
            alpha = gumbel_softmax(getattr(self, "alpha"), temperature=temp, hard=self.hard)

        latency_total = []

        latency, size = self.stem.forward_latency(size)
        latency_total.append(latency)

        for i, cell in enumerate(self.cells):
            latency, size = cell.forward_latency(size, alpha[i])
            latency_total.append(latency)

        latency, size = self.header.forward_latency(size)
        latency_total.append(latency)

        return sum(latency_total)


    def _loss(self, input, target, temp=1):

        logit = self(input, temp)
        loss = self._criterion(logit, target)

        return loss


    def _build_arch_parameters(self):
        num_ops = len(self.type)
        setattr(self, 'alpha', nn.Parameter(Variable(1e-3*torch.ones(sum(self.num_layer_list), num_ops).cuda(), requires_grad=True)))


        return {"alpha": self.alpha}


    def _reset_arch_parameters(self):
        num_ops = len(self.type)

        getattr(self, "alpha").data = Variable(1e-3*torch.ones(sum(self.num_layer_list), num_ops).cuda(), requires_grad=True)

    def clip(self):
        for line in getattr(self, "alpha"):
            max_index = line.argmax()
            line.data.clamp_(0, 1)
            if line.sum() == 0.0:
                line.data[max_index] = 1.0
            line.data.div_(line.sum())


if __name__ == '__main__':
    model = FBNet(num_classes=10)
    print(model)
