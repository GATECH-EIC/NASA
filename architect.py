import torch
import numpy as np
import sys
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
import torchcontrib
import numpy as np
from pdb import set_trace as bp
from thop import profile
# from operations import *
# from genotypes import PRIMITIVES


def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])


class Architect(object):

    def __init__(self, model, args):
        # self.network_momentum = args.momentum
        # self.network_weight_decay = args.weight_decay
        self.model = model
        self._args = args

        self.optimizer = torch.optim.Adam(list(self.model.module._arch_params.values()), lr=args.arch_learning_rate, betas=(0.5, 0.999))#, weight_decay=args.arch_weight_decay)
        
        self.alpha_weight = args.alpha_weight

        self.flops_weight = args.flops_weight

        self.flops_decouple = args.flops_decouple


        # self.latency_weight = args.latency_weight

        self.mode = args.mode


        self.offset = args.offset


        self.weight_optimizer = None

        print("architect initialized!")


    def set_weight_optimizer(self, weight_optimizer):
        self.weight_optimizer = weight_optimizer
        # print("one-level optimization!!")


    def step(self, input_valid, target_valid, config, temp=1):
        self.optimizer.zero_grad()

        if self.mode == 'proxy_hard' and self.offset:
            alpha_end_old = self.model.module._arch_params['alpha_end'].data.clone()
            alpha_middle_old = self.model.module._arch_params['alpha_middle'].data.clone()
            # print('alpha_old',alpha_old)
        
        # self.model.module.unused_modules_off(temp)

        if self.weight_optimizer is not None:
            self.weight_optimizer.zero_grad()
        
        if self._args.efficiency_metric == None:
            loss = self._backward_step(input_valid, target_valid, temp)
        if self._args.efficiency_metric == 'flops':
            loss, loss_flops = self._backward_step_flops(input_valid, target_valid, temp)

        # elif self._args.efficiency_metric == 'energy':
        #     loss, loss_energy = self._backward_step_energy(input_valid, target_valid, temp)

        # elif self._args.efficiency_metric == 'latency':
        #     loss, loss_latency = self._backward_step_latency(input_valid, target_valid, temp)

        # else:
        #     print('Wrong efficiency metric.')
        #     sys.exit()

        if self._args.arch_one_hot_loss_weight:
            prob_alpha = F.softmax(getattr(self.model.module, 'alpha'), dim=-1)
            loss += self._args.arch_one_hot_loss_weight * (torch.mean(- prob_alpha * torch.log(prob_alpha)))

        if self._args.arch_mse_loss_weight:
            prob_alpha = F.softmax(getattr(self.model.module, 'alpha'), dim=-1)
            loss += self._args.arch_mse_loss_weight * (torch.mean(-torch.pow((prob_alpha - 0.5), 2)))

        loss.backward()

        ## decouple the efficiency loss of alpha and beta
        if self._args.efficiency_metric == 'flops' and self.flops_weight > 0:
            loss_flops.backward()

        # elif self._args.efficiency_metric == 'latency' and self.latency_weight > 0:
        #     loss_latency.backward()

        # else:
        #     print('Wrong efficiency metric:', self._args.efficiency_metric)
        #     sys.exit()
        # self.optimizer = torch.optim.Adam(list(self.model.module._arch_params.values()), lr=self._args.arch_learning_rate, betas=(0.5, 0.999))
        # self.optimizer.zero_grad()
        # loss.backward()
        self.optimizer.step()
        # self.model.module.unused_modules_back(temp)
        self.optimizer.zero_grad()
            

        # update weight is one-level optimization
        if self.weight_optimizer is not None:
            nn.utils.clip_grad_norm_(self.model.parameters(), config.grad_clip)
            self.weight_optimizer.step()
            

        if self.mode == 'proxy_hard' and self.offset:
            alpha_end_new = self.model.module._arch_params['alpha_end'].data
            alpha_middle_new = self.model.module._arch_params['alpha_middle'].data
            # print(alpha_new)

            count_end = 0
            count_middle = 0
            for i, cell in enumerate(self.model.module.cells):
                if i < 3 or i > 18:
                    offset = torch.log(sum(torch.exp(alpha_end_old[count_end][cell.active_list])) / sum(torch.exp(alpha_end_new[count_end][cell.active_list])))
                    for active_op in cell.active_list:
                        self.model.module._arch_params['alpha_end'][count_end][active_op].data += offset.data
                    count_end += 1
                else:
                    offset = torch.log(sum(torch.exp(alpha_middle_old[count_middle][cell.active_list])) / sum(torch.exp(alpha_middle_new[count_middle][cell.active_list])))
                    for active_op in cell.active_list:
                        self.model.module._arch_params['alpha_middle'][count_middle][active_op].data += offset.data
                    count_middle += 1

        return loss

    def crossentropyloss(self,x,y):

        softmax_func=nn.Softmax(dim=1)
        soft_output=softmax_func(x)
        soft_output=torch.clamp(soft_output, 1e-4, 1, out=None)

        log_output=torch.log(soft_output)
        # print("log_output",log_output)

        #pytorch中关于NLLLoss的默认参数配置为：reducetion=True、size_average=True
        nllloss_func=nn.NLLLoss()
        nlloss_output=nllloss_func(log_output,y)
        return nlloss_output


    def _backward_step(self, input_valid, target_valid, temp=1):
        logit = self.model(input_valid, temp)
        # loss = self.model.module._criterion(logit, target_valid)
        loss = self.crossentropyloss(logit, target_valid)

        return loss
    

    def _backward_step_energy(self, input_valid, target_valid, temp=1):
        logit = self.model(input_valid, temp)
        loss = self.model.module._criterion(logit, target_valid)

        if self.energy_weight > 0:
            energy = self.model.module.forward_energy((3, self._args.image_height, self._args.image_width), temp)
        else:
            energy = 0
            
        self.energy_supernet = energy
        loss_energy = self.energy_weight * energy

        return loss, loss_energy


    def _backward_step_latency(self, input_valid, target_valid, temp=1):
        logit = self.model(input_valid, temp)
        loss = self.model.module._criterion(logit, target_valid)

        if self.latency_weight > 0:
            latency = self.model.module.forward_latency((3, self._args.image_height, self._args.image_width), temp)
        else:
            latency = 0

        self.latency_supernet = latency
        loss_latency = self.latency_weight * latency

        return loss, loss_latency


    def _backward_step_flops(self, input_valid, target_valid, temp=1):

        # TODO:
        # logit, kl_loss = self.model(input_valid, temp)
        logit = self.model(input_valid, temp)
        loss = self.model.module._criterion(logit, target_valid)

        if self.flops_weight > 0:
            if self.flops_decouple:
                flops_alpha = self.model.module.forward_flops((3, self._args.image_height, self._args.image_width), temp, alpha_only=True)
                flops_beta = self.model.module.forward_flops((3, self._args.image_height, self._args.image_width), temp, beta_only=True)
                flops = flops_alpha + flops_beta
            else:
                flops = self.model.module.forward_flops((3, self._args.image_height, self._args.image_width), temp)
        else:
            flops = 0

        self.flops_supernet = flops
        loss_flops = self.flops_weight * flops
        
        return loss, loss_flops


