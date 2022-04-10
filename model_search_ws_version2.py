import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.nn import init
from operations_ws_version2 import *
# import operations_ws
from torch.autograd import Variable
from genotypes import PRIMITIVES_OnlyConv, PRIMITIVES_AddAdd, PRIMITIVES_AddShift, PRIMITIVES_AddShiftAdd, PRIMITIVES_AddAll, PRIMITIVES_NoConv, PRIMITIVES_AddAdd_allconv
import numpy as np
from thop import profile
from matplotlib import pyplot as plt
from thop import profile
import math

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


class shared_parameters(nn.Module):
    def __init__(self, C_in, C_out, kernel, expansion=6):
        super(shared_parameters, self).__init__()

        self.conv1 = nn.Parameter(torch.randn(C_in*expansion, C_in, 1, 1))
        self.conv2 = nn.Parameter(torch.randn(C_in*expansion, 1, kernel, kernel))
        self.conv3 = nn.Parameter(torch.randn(C_out, C_in*expansion, 1, 1))

        # ############## initialization ######################
        init.kaiming_normal_(self.conv1, mode='fan_out')
        init.kaiming_normal_(self.conv2, mode='fan_out')
        init.kaiming_normal_(self.conv3, mode='fan_out')
    
    def forward(self, x):
        # for key, item in self.shared_weight.items():
        #     x = x * item
        return x

# FIXME: 3 places(2 in model_search; 1 in model_infer)

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
            # TODO:
            self.type = PRIMITIVES_AddAdd
            # if layer_id < 4 or layer_id > 19:
            #     self.type = PRIMITIVES_AddAdd_allconv
            # else:
            #     self.type = PRIMITIVES_AddAdd
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
            # ########### both channel and kernel weight sharing #########
            # if (primitive=='k5_e6' or primitive=='add_k5_e6' or primitive=='shift_k5_e6' or primitive=='shiftadd_k5_e6' or primitive=='skip'):
                if 'k3' in primitive:
                    shared_parameter = shared_parameters(C_in, C_out, kernel=3)
                elif 'k5' in primitive:
                    shared_parameter = shared_parameters(C_in, C_out, kernel=5)
                else: 
                    shared_parameter = None
                op = OPS[primitive](C_in, C_out, layer_id, stride, shared_parameter)
                self._ops.append(op)
        # print(self._ops)
        self.register_buffer('active_list', torch.tensor(list(range(len(self._ops)))))


    # ############## chanell-wise weight sharing ##################
    def forward(self, x, alpha, alpha_param=None, update_arch=True, full_kernel=False, full_channel=False, cand=None):
        # print('ok')
        # int: force #channel; tensor: arch_ratio; float(<=1): force width
        result = 0
        kl_loss =0

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
                if (update_arch == False):
                    if (full_channel == False):
                        np.random.shuffle(rank.cpu().detach().numpy())
                    else:
                        index = []
                        while len(index)!= self.act_num:
                            id = np.random.randint(len(self._ops)) 
                            if id not in index:
                                index.append(id)
                    # print(rank)
            
                self.set_active_list(rank[:self.act_num])
                # print('ok')
                # print(self.active_list)

                alpha = F.softmax(alpha_param[rank[:self.act_num]], dim=-1)
        
                for i in range(self.act_num):
                    # print(i)
                   
                    # print(self.type[rank[i]])
                    if full_channel==False:
                        if(rank[i]==(len(self.type)-1)):
                            result = result + self._ops[-1](x)[0] * alpha[i]
                            kl_loss =  kl_loss + self._ops[-1](x)[1] * alpha[i]
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
                            result = result + self._ops[type](x,ratio)[0] * alpha[i]
                            kl_loss =  kl_loss + self._ops[type](x,ratio)[1] * alpha[i]
                    else:
                        result = result + self._ops[index[i]](x)[0] * alpha[i]
                        kl_loss =  kl_loss + self._ops[index[i]](x)[1] * alpha[i]
                    # result = result + self._ops[rank[i]](x) * ((0-alpha[i]).detach() + alpha[i])
            else:
                self.set_active_list(cand)
                if(cand==(len(self.type)-1)):
                    result = result + self._ops[-1](x)[0]
                    kl_loss =  kl_loss + self._ops[-1](x)[1] 
                else:
                    type = cand // 3
                    ratio = cand % 3
                    if (ratio==0):
                        ratio=6
                    if (ratio==1):
                        ratio=2
                    if (ratio==2):
                        ratio=1
                    result = result + self._ops[type](x,ratio)[0]
                    kl_loss =  kl_loss + self._ops[type](x,ratio)[1] 
        else:
            print('Wrong search mode:', self.mode)
            sys.exit()

        return result, kl_loss
    
    # ######## both channel and kernel weight sharing ############
    # def forward(self, x, alpha, alpha_param=None, update_arch=True, full_kernel=False, full_channel=False, cand=None):
    #     # int: force #channel; tensor: arch_ratio; float(<=1): force width
    #     result = 0

    #     if self.mode == 'soft':
    #         for i, (w, op) in enumerate(zip(alpha, self._ops)):
    #             result = result + op(x) * w 

    #         self.set_active_list(list(range(len(self._ops))))

    #     elif self.mode == 'proxy_hard':
    #         if (cand==None):
    #             assert alpha_param is not None
    #             rank = alpha.argsort(descending=True)
    #             if (update_arch == False):
    #                 if (full_channel == False and full_kernel==False):
    #                     np.random.shuffle(rank.cpu().detach().numpy())
    #                 else:
    #                     index = []
    #                     while len(index)!= self.act_num:
    #                         id = np.random.randint(len(self._ops)) 
    #                         if id not in index:
    #                             index.append(id)
            
    #             self.set_active_list(rank[:self.act_num])

    #             alpha = F.softmax(alpha_param[rank[:self.act_num]], dim=-1)
        
    #             for i in range(self.act_num):
    #                 # print(i)
                   
    #                 # print(self.type[rank[i]])
    #                 if (full_channel == False and full_kernel==False):
    #                     if(rank[i]==(len(self.type)-1)):
    #                         result = result + self._ops[-1](x) * alpha[i]
    #                     else:
    #                         # type = rank[i]//3
    #                         # ratio = rank[i]%3
    #                         # if (ratio==0):
    #                         #     ratio=6
    #                         # if (ratio==1):
    #                         #     ratio=2
    #                         # if (ratio==2):
    #                         #     ratio=1
    #                         type = rank[i]//6
    #                         ratio = rank[i]%3
    #                         kernel = rank[i]%2
    #                         if (ratio==0):
    #                             ratio=6
    #                         if (ratio==1):
    #                             ratio=2
    #                         if (ratio==2):
    #                             ratio=1
    #                         if (kernel==0):
    #                             kernel=3
    #                         if (kernel==1):
    #                             kernel=None
    #                         result = result + self._ops[type](x,ratio,kernel) * alpha[i]
    #                 else:
    #                     if full_kernel == False:
    #                         # print(index[i])
    #                         if (index[i]==(len(self._ops)-1)):
    #                             result = result + self._ops[index[i]](x) * alpha[i]
    #                         else:
    #                             kernel = np.random.randint(0,2)
    #                             # print(kernel)
    #                             if kernel==0:
    #                                 kernel=None
    #                             else:
    #                                 kernel=3
    #                             # print(self._ops[index[i]])
    #                             result = result + self._ops[index[i]](x,kernel=kernel) * alpha[i]
    #                     else:
    #                         result = result + self._ops[index[i]](x) * alpha[i]
    #                 # result = result + self._ops[rank[i]](x) * ((0-alpha[i]).detach() + alpha[i])
    #         else:
    #             self.set_active_list(cand)
    #             if(cand==(len(self.type)-1)):
    #                 result = result + self._ops[-1](x)
    #             else:
    #                 type = cand // 6
    #                 ratio = cand % 3
    #                 kernel = cand % 2
    #                 if (ratio==0):
    #                     ratio=6
    #                 if (ratio==1):
    #                     ratio=2
    #                 if (ratio==2):
    #                     ratio=1
    #                 if (kernel==0):
    #                     kernel=3
    #                 if (kernel==1):
    #                     kernel=None
    #                 result = result + self._ops[type](x,ratio,kernel)

    #     else:
    #         print('Wrong search mode:', self.mode)
    #         sys.exit()
    #     # else:
    #     #     rank = alpha.argsort(descending=True)
    #     #     
    #     #     result = result + self._ops[rank[0]](x) * alpha[0]
    #     #     for i in range(1,self.act_num):
    #     #         result = result + self._ops[rank[i]](x) * alpha[i]
        
    #     # print(result)
    #     return result

        
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
            self.type = PRIMITIVES_AddAdd
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
                try:
                    init.kaiming_normal_(m.weight, mode='fan_out')
                    if m.bias is not None:
                        init.constant_(m.bias, 0)
                except:
                    pass
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


    def forward(self, input, temp=1, update_arch=True, full_channel=False, full_kernel=False, cand=None):
        # print('ok!')
        if self.sample_func == 'softmax':
            alpha = F.softmax(getattr(self, "alpha"), dim=-1)
        else:
            alpha = gumbel_softmax(getattr(self, "alpha"), temperature=temp, hard=self.hard)
    
        out = self.stem(input)

        if (cand==None):
            # print('ok')
            for i, cell in enumerate(self.cells):
                # print(i)
                out, kl_loss = cell(out, alpha[i], getattr(self, "alpha")[i], update_arch, full_kernel, full_channel, cand)
        else:
            for i, cell in enumerate(self.cells):
                out, kl_loss = cell(out, alpha[i], getattr(self, "alpha")[i], update_arch, full_kernel, full_channel, cand[i])
        # print(out)

        
        # TODO:
        out = self.fc(self.avgpool(self.header(out)).view(out.size(0), -1))
        return out, kl_loss
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
            # TODO:
            # print(self.type[op_idx_list[i]], end=' ')
            # if i < 3 or i > 18:
            #     self.type = PRIMITIVES_AddAdd_allconv
            # else:
            #     self.type = PRIMITIVES_AddAdd
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
