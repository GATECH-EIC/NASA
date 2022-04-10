import sys
import os
from numpy.lib.twodim_base import triu_indices_from
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.nn import init
from operations_ws import *
import operations_ws
from torch.autograd import Variable
from genotypes import PRIMITIVES_OnlyConv, PRIMITIVES_AddAdd, PRIMITIVES_AddShift, PRIMITIVES_AddShiftAdd, PRIMITIVES_AddAll, PRIMITIVES_NoConv, PRIMITIVES_AddAdd_allconv
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
            # self.type = PRIMITIVES_AddAdd
            # TODO:
            if layer_id < 4 or layer_id > 19:
                self.type = PRIMITIVES_AddAdd_allconv
            else:
                self.type = PRIMITIVES_AddAdd
        if(self.search_space=='AddShift'):
            self.type = PRIMITIVES_AddShift
        if(self.search_space=='AddShiftAdd'):
            self.type = PRIMITIVES_AddShiftAdd
        if(self.search_space=='AddAll'):
            self.type = PRIMITIVES_AddAll
        if(self.search_space=='NoConv'):
            self.type = PRIMITIVES_NoConv

        for primitive in self.type:
            # ######## channel weight sharing ################
            if (primitive=='k3_e6' or primitive=='k5_e6' or primitive=='add_k3_e6' or primitive=='add_k5_e6' or primitive=='shift_k3_e6' or primitive=='shift_k5_e6' or 
            primitive=='shiftadd_k3_e6' or primitive=='shiftadd_k5_e6' or 
            primitive=='skip'):
                op = OPS[primitive](C_in, C_out, layer_id, stride)
                self._ops.append(op)
        # print(self._ops)
        self.register_buffer('active_list', torch.tensor(list(range(len(self._ops)))))


    # ############## chanell-wise weight sharing ##################
    # def forward(self, x, alpha, alpha_param=None, update_arch=True, full_kernel=False, full_channel=False, all_conv=False, all_add=False, mix=False, cand=None):
    #     # print('ok')
    #     # int: force #channel; tensor: arch_ratio; float(<=1): force width
    #     result = 0

    #     if self.mode == 'soft':
    #         for i, (w, op) in enumerate(zip(alpha, self._ops)):
    #             result = result + op(x) * w 

    #         self.set_active_list(list(range(len(self._ops))))

    #     elif self.mode == 'proxy_hard':
    #         # print('ok')
    #         # print('cand',cand)
    #         if (cand==None):
    #             # print('ok')
    #             assert alpha_param is not None
                
    #             rank = alpha.argsort(descending=True)
    #             if (update_arch == False):
    #                 if full_channel == True:
    #                     if (mix==True) or (self.layer_id < 4 or self.layer_id > 19):
    #                         index = []
    #                         while len(index)!= self.act_num:
    #                             id = np.random.randint(len(self._ops)) 
    #                             if id not in index:
    #                                 index.append(id)
    #                     elif all_conv == True:
    #                         index = []
    #                         conv = [0,1,4]
    #                         while len(index)!= self.act_num:
    #                             id = random.choice(conv)
    #                             if id not in index:
    #                                 index.append(id)
    #                             # print(index)
    #                     elif all_add == True:
    #                         index = []
    #                         add = [2,3]
    #                         while len(index)!= self.act_num:
    #                             id = random.choice(add)
    #                             if id not in index:
    #                                 index.append(id) 
    #                 else:
    #                     np.random.shuffle(rank.cpu().detach().numpy())
    #                 # print(rank)
            
    #             self.set_active_list(rank[:self.act_num])
    #             # print('ok')
    #             # print(self.active_list)

    #             alpha = F.softmax(alpha_param[rank[:self.act_num]], dim=-1)
        
    #             for i in range(self.act_num):
    #                 # print(i)
                   
    #                 # print(self.type[rank[i]])
    #                 if full_channel == False:
    #                     type = rank[i] // 3 
    #                     ratio = rank[i] % 3
    #                     if (ratio==0):
    #                         ratio=6
    #                     if (ratio==1):
    #                         ratio=2
    #                     if (ratio==2):
    #                         ratio=1
    #                     # print(self._ops[type],ratio)
    #                     result = result + self._ops[type](x,ratio) * alpha[i]
    #                 else:
    #                     result = result + self._ops[index[i]](x) * alpha[i]
    #                 # result = result + self._ops[rank[i]](x) * ((0-alpha[i]).detach() + alpha[i])
    #         else:
    #             self.set_active_list(cand)
    #             if(cand==(len(self.type)-1)):
    #                 result = result + self._ops[-1](x)
    #             else:
    #                 type = cand // 3
    #                 ratio = cand % 3
    #                 if (ratio==0):
    #                     ratio=6
    #                 if (ratio==1):
    #                     ratio=2
    #                 if (ratio==2):
    #                     ratio=1
    #                 result = result + self._ops[type](x,ratio)

    #     else:
    #         print('Wrong search mode:', self.mode)
    #         sys.exit()

    #     return result

    def forward(self, x, alpha, alpha_param=None, update_arch=True, full_kernel=False, full_channel=False, all_conv=False, all_add=False, mix=True, cand=None):
        # print('ok')
        # int: force #channel; tensor: arch_ratio; float(<=1): force width
        result = 0

        if self.mode == 'soft':
            for i, (w, op) in enumerate(zip(alpha, self._ops)):
                result = result + op(x) * w 

            self.set_active_list(list(range(len(self._ops))))

        elif self.mode == 'proxy_hard':
            # print('ok')
            # print('cand',cand)
            if (cand==None):
                # print('ok')
                assert alpha_param is not None
                
                rank = alpha.argsort(descending=True)
                if mix==False:
                    index = []
                    # print()
                    if (self.layer_id < 4 or self.layer_id > 19):
                        if full_channel == True: 
                            op = [0,1,2]
                        else:
                            op = [0,1,2,3,4,5,6]
                    elif all_conv == True:
                        if full_channel == True: 
                            op = [0,1,4]
                        else:
                            op = [0,1,2,3,4,5,12]
                    elif all_add == True:
                        if full_channel == True: 
                            op = [2,3]
                        else:
                            op = [3,7,8,9,10,11]
                    while len(index)!= self.act_num:
                        id = random.choice(op)
                        if id not in index:
                            index.append(id)
                
                else:
                    np.random.shuffle(rank.cpu().detach().numpy())
                    # print(rank)
            
                self.set_active_list(rank[:self.act_num])
                # print('ok')
                # print(self.active_list)

                alpha = F.softmax(alpha_param[rank[:self.act_num]], dim=-1)
        
                for i in range(self.act_num):
                    # print(i)
                   
                    # print(self.type[rank[i]])
                    if full_channel == True:
                        result = result + self._ops[index[i]](x) * alpha[i]
                    else:
                        if mix == False:
                            type = index[i] // 3 
                            ratio = index[i] % 3
                        else:
                            type = rank[i] // 3 
                            ratio = rank[i] % 3
                        if (ratio==0):
                            ratio=6
                        if (ratio==1):
                            ratio=2
                        if (ratio==2):
                            ratio=1
                        # print(self._ops[type],ratio)
                        result = result + self._ops[type](x,ratio) * alpha[i]
            else:
                self.set_active_list(cand)
                if(cand==(len(self.type)-1)):
                    result = result + self._ops[-1](x)
                else:
                    type = cand // 3
                    ratio = cand % 3
                    if (ratio==0):
                        ratio=6
                    if (ratio==1):
                        ratio=2
                    if (ratio==2):
                        ratio=1
                    result = result + self._ops[type](x,ratio)

        else:
            print('Wrong search mode:', self.mode)
            sys.exit()

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

    
    def forward_flops(self, size, alpha):
        # int: force #channel; tensor: arch_ratio; float(<=1): force width
        result = 0

        op_id = alpha.argsort(descending=True)[0]
        if(op_id==(len(self.type)-1)):
            flops, size_out = self._ops[-1].forward_flops(size)
        else:
            type = op_id // 3 
            ratio = op_id % 3
            if (ratio==0):
                ratio=1
            if (ratio==1):
                ratio=3
            if (ratio==2):
                ratio=6
            # print(self._ops[type],ratio)
            flops, size_out = self._ops[type].forward_flops(size, ratio)
        # print(alpha)
        # print("op_id:",op_id)
        # flops, size_out = self._ops[op_id].forward_flops(size)
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
            self.type1 = PRIMITIVES_AddAdd_allconv
            self.type2 = PRIMITIVES_AddAdd
        if(self.search_space=='AddShift'):
            self.type = PRIMITIVES_AddShift
        if(self.search_space=='AddShiftAdd'):
            self.type = PRIMITIVES_AddShiftAdd
        if(self.search_space=='AddAll'):
            self.type = PRIMITIVES_AddAll
        if(self.search_space=='NoConv'):
            self.type = PRIMITIVES_NoConv

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


    def forward(self, input, temp=1, update_arch=True, full_channel=False, full_kernel=False, all_conv=False, all_add=False, mix=True, cand=None):
        # print('ok!')
        if self.sample_func == 'softmax':
            alpha_end = F.softmax(getattr(self, "alpha_end"), dim=-1)
            alpha_middle = F.softmax(getattr(self, "alpha_middle"), dim=-1)
        else:
            alpha_end = gumbel_softmax(getattr(self, "alpha_end"), temperature=temp, hard=self.hard)
            alpha_middle = gumbel_softmax(getattr(self, "alpha_middle"), temperature=temp, hard=self.hard)
    
        out = self.stem(input)

        if (cand==None):
            count_end = 0
            count_middle = 0
            for i, cell in enumerate(self.cells):
                if i < 3 or i > 18:
                    out = cell(out, alpha_end[count_end], getattr(self, "alpha_end")[count_end], update_arch, full_kernel, full_channel, all_conv, all_add, mix, cand)
                    count_end += 1
                else:
                    out = cell(out, alpha_middle[count_middle], getattr(self, "alpha_middle")[count_middle], update_arch, full_kernel, full_channel, all_conv, all_add, mix, cand)
                    count_middle += 1
        else:
            for i, cell in enumerate(self.cells):
                out = cell(out, alpha[i], getattr(self, "alpha")[i], update_arch, full_kernel, full_channel, all_conv, all_add, mix, cand[i])
        # print(out)

        
        # TODO:
        out = self.fc(self.avgpool(self.header(out)).view(out.size(0), -1))
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

    def show_arch(self, alpha_end=None, alpha_middle=None, cands=None):
        # if self.sample_func == 'softmax':
        #     alpha = F.softmax(getattr(self, "alpha"), dim=-1)
        # else:
        #     alpha = gumbel_softmax(getattr(self, "alpha"), temperature=temp, hard=self.hard)
        if alpha_end != None and alpha_middle != None:
            op_idx_list_end = F.softmax(alpha_end, dim=-1).argmax(-1)
            op_idx_list_middle = F.softmax(alpha_middle, dim=-1).argmax(-1)
        else:
            op_idx_list = cands

        count_end = 0
        count_middle = 0
        for i, _ in enumerate(self.cells):
            # TODO:
            if i < 3 or i > 18:
                print(self.type1[op_idx_list_end[count_end]], end=' ')
                count_end += 1
            else:
                print(self.type2[op_idx_list_middle[count_middle]], end=' ')
                count_middle += 1


    def forward_flops(self, size, temp=1):
        if self.sample_func == 'softmax':
            alpha = F.softmax(getattr(self, "alpha"), dim=-1)
        else:
            alpha_end = gumbel_softmax(getattr(self, "alpha_end"), temperature=temp, hard=self.hard)
            alpha_middle = gumbel_softmax(getattr(self, "alpha_middle"), temperature=temp, hard=self.hard)

        flops_total = []

        flops, size = self.stem.forward_flops(size)
        flops_total.append(flops)

        count_end = 0
        count_middle = 0
        for i, cell in enumerate(self.cells):
            if i < 3 or i > 18:
                flops, size = cell.forward_flops(size, alpha_end[count_end])
                count_end += 1
            else:
                flops, size = cell.forward_flops(size, alpha_middle[count_middle])
                count_middle += 1
            flops_total.append(flops)

        flops, size = self.header.forward_flops(size)
        flops_total.append(flops)

        return sum(flops_total)


    def _loss(self, input, target, temp=1):

        logit = self(input, temp)
        loss = self._criterion(logit, target)

        return loss


    def _build_arch_parameters(self):
        num_ops1 = len(self.type1)
        num_ops2 = len(self.type2)

        setattr(self, 'alpha_end', nn.Parameter(Variable(1e-3*torch.ones(6, num_ops1).cuda(), requires_grad=True)))
        setattr(self, 'alpha_middle', nn.Parameter(Variable(1e-3*torch.ones(sum(self.num_layer_list)-6, num_ops2).cuda(), requires_grad=True)))

        return {"alpha_end": self.alpha_end, "alpha_middle": self.alpha_middle}


    def _reset_arch_parameters(self):
        # num_ops = len(self.type)
        num_ops1 = len(self.type1)
        num_ops2 = len(self.type2)

        getattr(self, 'alpha_end', nn.Parameter(Variable(1e-3*torch.ones(6, num_ops1).cuda(), requires_grad=True)))
        getattr(self, 'alpha_middle', nn.Parameter(Variable(1e-3*torch.ones(sum(self.num_layer_list)-6, num_ops2).cuda(), requires_grad=True)))

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
