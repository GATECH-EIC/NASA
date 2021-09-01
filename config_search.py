# encoding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import sys
import time
import numpy as np
import genotypes
from easydict import EasyDict as edict

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

add_path('hw_diff_final')
add_path('fpga_nips')

C = edict()
config = C
cfg = C

C.seed = 2

"""please config ROOT_dir and user when u first using"""
C.repo_name = 'DNA'

C.world_size = 1  # num of nodes
C.multiprocessing_distributed = False
C.rank = 0  # node rank
C.dist_backend = 'nccl'
C.dist_url = 'tcp://eic-2019gpu5.ece.rice.edu:10001'  # url used to set up distributed training
# C.dist_url = 'tcp://127.0.0.1:10001'

C.gpu = None

""""" set datasets """""
# TODO:
C.dataset = 'cifar100'

if 'cifar' in C.dataset:
    C.dataset_path = "/media/shared-corpus/CIFAR100"

    # if C.dataset == 'cifar10':
    #     C.num_classes = 10
    # elif C.dataset == 'cifar100':
    #     C.num_classes = 100
    # else:
    #     print('Wrong dataset.')
    #     sys.exit()

    """Image Config"""

    C.num_train_imgs = 50000
    C.num_eval_imgs = 10000

    """ Settings for network, this would be different for each kind of model"""
    C.bn_eps = 1e-5
    C.bn_momentum = 0.1

    """Train Config"""

    C.opt = 'Sgd'

    C.momentum = 0.9
    C.weight_decay = 5e-4

    C.betas=(0.5, 0.999)
    C.num_workers = 8

    """ Search Config """
    C.grad_clip = 5

    C.pretrain = 'ckpt/search'
    
    """ Supernet Config"""
    
    # C.num_layer_list = [1, 1, 1, 1, 1, 1, 1] 
    C.num_layer_list = [1, 4, 4, 4, 4, 4, 1] 
    C.num_channel_list = [16, 24, 32, 64, 112, 184, 352]
    C.stride_list = [1, 2, 2, 2, 1, 2, 1]

    C.stem_channel = 16
    C.header_channel = 1504

    C.enable_skip = True
    if not C.enable_skip:
        if 'skip' in genotypes.PRIMITIVES:
            genotypes.PRIMITIVES.remove('skip')



    C.perturb_alpha = False
    C.epsilon_alpha = 0.3


    C.sample_func = 'gumbel_softmax' # sampling function used for DNAS
    C.temp_init = 5
    C.temp_decay = 0.975

    ## Gumbel Softmax settings for operator
    C.mode = 'proxy_hard'  # sampling methods used for DNAS. 'proxy_hard' is the method used in ProxylessNas, 'soft' is the method used in FBNet. 
    if C.mode == 'soft':
        C.hard = False
    else:
        C.hard = True

    C.offset = True and C.mode == 'proxy_hard'
    
    # TODO:
    # C.act_num = 2 # number of active paths used during each update in search
    C.act_num = 3

    #TODO:
    C.pretrain_epoch = 110
    C.search_space = 'OnlyConv'
    # C.pretrain_epoch = 1
    C.pretrain_aline = True

    if C.pretrain_aline:
        C.pretrain_mode = C.mode
        C.pretrain_act_num = C.act_num
    else:
        C.pretrain_mode = 'soft'
        C.pretrain_act_num = 1

    C.arch_one_hot_loss_weight = None
    C.arch_mse_loss_weight = None

    C.num_sample = 10

    C.update_hw_freq = 5

    ########################################

    C.batch_size = 32
    C.niters_per_epoch = C.num_train_imgs // 2 // C.batch_size
    C.image_height = 32 
    C.image_width = 32

    
    # C.nepochs = 90 + C.pretrain_epoch
    # C.nepochs = 1 + C.pretrain_epoch
    C.eval_epoch = 1

    C.lr_schedule = 'cosine'
    
    
    C.lr = 0.05
    # C.lr = 0.00
    # linear 
    C.decay_epoch = 20
    # exponential
    C.lr_decay = 0.97
    # multistep
    C.milestones = [50, 100, 200]
    C.gamma = 0.1
    # cosine
    C.learning_rate_min = 0.001

    ########################################

    C.train_portion = 0.5  # 0.8

    C.unrolled = False
   
    C.arch_learning_rate = 3e-4
    # C.arch_learning_rate = 5e-4
    
    C.arch_update_frec = 1

    # hardware cost
    C.efficiency_metric = None # 'flops'
    assert C.efficiency_metric == 'flops' or C.efficiency_metric == 'latency' or C.efficiency_metric == 'energy' or C.efficiency_metric == None
    C.hw_platform_path = 'fbnet/edgegpu/' # path to the folder containing .npy file for efficiency metric

    # hardware cost weighted coefficients
    C.alpha_weight = 1

    # latency, customized for single-path FPGA predictor
    C.latency_weight = 1e-10  # The weight coefficient to add the hardward-cost in the loss
    C.fps_max = 100 # targetting FPS range during search
    C.fps_min = 90

    # FLOPs
    C.flops_mode = 'single_path' # 'single_path', 'multi_path'

    C.flops_weight = 0
    C.flops_max = 3e8
    C.flops_min = 5e7
    C.flops_decouple = False

elif 'imagenet' in C.dataset:
    C.dataset_path = "/media/HardDisk1/datadisk/imagenet"

    C.num_classes = 100

    """Image Config"""

    # C.num_train_imgs = 50000
    # C.num_eval_imgs = 10000

    """ Settings for network, this would be different for each kind of model"""
    C.bn_eps = 1e-5
    C.bn_momentum = 0.1

    """Train Config"""

    C.opt = 'Sgd'

    C.momentum = 0.9
    C.weight_decay = 5e-4

    C.betas=(0.5, 0.999)
    C.num_workers = 8

    """ Search Config """
    C.grad_clip = 5

    C.pretrain = 'ckpt/search'
    
    """ Supernet Config"""
    
    # C.num_layer_list = [1, 1, 1, 1, 1, 1, 1] 
    C.num_layer_list = [1, 4, 4, 4, 4, 4, 1] 
    C.num_channel_list = [16, 24, 32, 64, 112, 184, 352]
    C.stride_list = [1, 2, 2, 2, 1, 2, 1]

    C.stem_channel = 16
    C.header_channel = 1504

    C.enable_skip = True
    if not C.enable_skip:
        if 'skip' in genotypes.PRIMITIVES:
            genotypes.PRIMITIVES.remove('skip')


    C.perturb_alpha = False
    C.epsilon_alpha = 0.3

    C.sample_func = 'gumbel_softmax' # sampling function used for DNAS
    C.temp_init = 5
    C.temp_decay = 0.956

    ## Gumbel Softmax settings for operator
    C.mode = 'proxy_hard'  # sampling methods used for DNAS. 'proxy_hard' is the method used in ProxylessNas, 'soft' is the method used in FBNet. 
    if C.mode == 'soft':
        C.hard = False
    else:
        C.hard = True

    C.offset = True and C.mode == 'proxy_hard'
    
    # TODO:
    # C.act_num = 2 # number of active paths used during each update in search
    C.act_num = 3

    #TODO:
    C.pretrain_epoch = 30
    C.search_space = 'OnlyConv'
    # C.pretrain_epoch = 1
    C.pretrain_aline = True

    if C.pretrain_aline:
        C.pretrain_mode = C.mode
        C.pretrain_act_num = C.act_num
    else:
        C.pretrain_mode = 'soft'
        C.pretrain_act_num = 1

    C.arch_one_hot_loss_weight = None
    C.arch_mse_loss_weight = None

    C.num_sample = 10

    C.update_hw_freq = 5

    ########################################

    C.batch_size = 32
    # C.niters_per_epoch = C.num_train_imgs // 2 // C.batch_size
    C.image_height = 224 
    C.image_width = 224

    
    # C.nepochs = 90 + C.pretrain_epoch
    # C.nepochs = 1 + C.pretrain_epoch
    C.eval_epoch = 1

    C.lr_schedule = 'cosine'
    
    
    C.lr = 0.05
    # C.lr = 0.00
    # linear 
    C.decay_epoch = 20
    # exponential
    C.lr_decay = 0.97
    # multistep
    C.milestones = [50, 100, 200]
    C.gamma = 0.1
    # cosine
    C.learning_rate_min = 0.001

    ########################################

    C.train_portion = 0.8  # 0.8

    C.unrolled = False
   
    C.arch_learning_rate = 3e-4
    # C.arch_learning_rate = 5e-4
    
    C.arch_update_frec = 1

    # hardware cost
    C.efficiency_metric = 'flops' # 'flops'
    assert C.efficiency_metric == 'flops' or C.efficiency_metric == 'latency' or C.efficiency_metric == 'energy'
    C.hw_platform_path = 'fbnet/edgegpu/' # path to the folder containing .npy file for efficiency metric

    # hardware cost weighted coefficients
    C.alpha_weight = 1

    # latency, customized for single-path FPGA predictor
    C.latency_weight = 1e-10  # The weight coefficient to add the hardward-cost in the loss
    C.fps_max = 100 # targetting FPS range during search
    C.fps_min = 90

    # FLOPs
    C.flops_mode = 'single_path' # 'single_path', 'multi_path'

    C.flops_weight = 1e-10
    C.flops_max = 3e8
    C.flops_min = 2e8
    C.flops_decouple = False


else:
    print('Wrong dataset.')
    sys.exit()
