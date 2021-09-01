import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from operations import *
import operations
from torch.autograd import Variable
from genotypes import PRIMITIVES_OnlyConv, PRIMITIVES_AddAdd, PRIMITIVES_AddShift, PRIMITIVES_AddShiftAdd, PRIMITIVES_AddAll, PRIMITIVES_NoConv
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

        for primitive in self.type:
            op = OPS[primitive](C_in, C_out, layer_id, stride)
            self._ops.append(op)

        self.register_buffer('active_list', torch.tensor(list(range(len(self._ops)))))


    def forward(self, x, rank, alpha_param=None):
        # int: force #channel; tensor: arch_ratio; float(<=1): force width
        result = 0

        if self.mode == 'soft':
            for i, (w, op) in enumerate(zip(alpha, self._ops)):
                result = result + op(x) * w 

            self.set_active_list(list(range(len(self._ops))))

        elif self.mode == 'proxy_hard':
            assert alpha_param is not None
            # print(rank)
            # rank = alpha.argsort(descending=True)
            self.set_active_list(rank[:self.act_num])

            alpha = F.softmax(alpha_param[rank[:self.act_num]], dim=-1)
            
            # print("self.act_num, len(rank)",self.act_num, len(rank))

            # TODO: disable grad
            # for i in range(self.act_num, len(rank)):
            # for i in range(0, len(rank)):
            #     # self._ops[rank[i]].parameters.requires_grad = False
            #     # freeze(self._ops[rank[i]])
            #     self._ops[rank[i]].requires_grad_(False)

            # for i in range(0, self.act_num):
            #     self._ops[rank[i]].requires_grad_(True)
                # Activate(self._ops[rank[i]])
            # print(self._ops[rank[0]])
            result = result + self._ops[rank[0]](x) * ((1-alpha[0]-0.1).detach() + alpha[0])
            
            # result = result + self._ops[rank[0]](x)
           
            # TODO:
            for i in range(1,self.act_num):
                result = result + self._ops[rank[i]](x) * ((0-alpha[i]+0.1).detach() + alpha[i])
            #     print(result)
                # result = result + self._ops[rank[i]](x) 
                # result = result + self._ops[rank[i]](x)
                # print(self._ops[rank[i]](x).type())
        else:
            print('Wrong search mode:', self.mode)
            sys.exit()
        
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
        self._redundant_modules = None
        
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


    def forward(self, input, temp=1):
        if self.sample_func == 'softmax':
            alpha = F.softmax(getattr(self, "alpha"), dim=-1)
        else:
            alpha = gumbel_softmax(getattr(self, "alpha"), temperature=temp, hard=self.hard)
        # _ , total_rank = self.redundant_modules(temp)
    
        out = self.stem(input)
        
        # print("latter rank", self.total_rank)
        for i, cell in enumerate(self.cells):
            out = cell(out, self.total_rank[i], getattr(self, "alpha")[i])
            # print(i)
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

    # @property
    # def redundant_modules(self, temp):
    #     self.total_rank =[]
    #     # self.module_list = []
    #     # ################ create alpha ##################
    #     if self.sample_func == 'softmax':
    #         alpha = F.softmax(getattr(self, "alpha"), dim=-1)
    #     else:
    #         alpha = gumbel_softmax(getattr(self, "alpha"), temperature=temp, hard=self.hard)
        
    #     # ################ save redundant_modules ################
    #     if self._redundant_modules is None:
    #         for i, cell in enumerate(self.cells):
    #             m = {}
    #             rank = alpha[i].argsort(descending=True)
    #             self.total_rank.append(rank)
    #         # for j, operator in enumerate(cell):
    #             for j in range(self.act_num, len(rank)):
    #                 # m[j] = cell._ops[rank[j]]
    #                 m[j] = rank[j]
    #             self.module_list.append(m)
    #         # self._redundant_modules = module_list
    #     # return self.module_list, self.total_rank


    def unused_modules_off(self, temp):
        self.total_rank =[]
        self._unused_modules = []
        # ################ create alpha ##################
        if self.sample_func == 'softmax':
            alpha = F.softmax(getattr(self, "alpha"), dim=-1)
        else:
            alpha = gumbel_softmax(getattr(self, "alpha"), temperature=temp, hard=self.hard)

        # ################ save redundant_modules ################
        for i, cell in enumerate(self.cells):
            unused = {}
            # rank = self.total_rank[i]
            rank = alpha[i].argsort(descending=True)
            self.total_rank.append(rank)
            for j in range(self.act_num, len(rank)):
                unused[j] = cell._ops[rank[j]]
                cell._ops[rank[j]] = None
            self._unused_modules.append(unused)

        # print("original rank", self.total_rank)

        # print("rank one", total_rank)       

    def unused_modules_back(self, temp):
        
        # # module_list, _ = self.redundant_modules(temp)
        # if self._unused_modules is None:
        #     return
        # for m, unused in zip(self.module_list, self._unused_modules):
        #     for i in unused:
        #         m[i] = unused[i]
        # self._unused_modules = None
        

        if self._unused_modules is None:
            return
        # for i, cell in enumerate(self.cells):
        i = 0
        for cell, unused in zip(self.cells, self._unused_modules):
            rank = self.total_rank[i]
            for j in unused:
                cell._ops[rank[j]] = unused[j]
            i += 1
        self._unused_modules = None
        
        del i



if __name__ == '__main__':
    model = FBNet(num_classes=10)
    # print(model)
