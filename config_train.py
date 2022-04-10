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

# TODO:
C.dataset = 'cifar10'

if 'cifar' in C.dataset:
    """Data Dir and Weight Dir"""
    # TODO:
    C.dataset_path = "/media/HardDisk1/cifar/CIFAR10"

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
    C.num_workers = 4


    """ Search Config """
    C.grad_clip = 5
    # TODO:
    C.pretrain = '/media/HardDisk1/shihh/NAS/ckpt/CIFAR10_AddShiftAdd_scratch_70_int16'
    C.search_space = 'AddShiftAdd'

    C.num_layer_list = [1, 4, 4, 4, 4, 4, 1]
    C.num_channel_list = [16, 24, 32, 64, 112, 184, 352]
    C.stride_list = [1, 1, 2, 2, 1, 2, 1]

    C.stem_channel = 16
    C.header_channel = 1504

    C.use_hswish = False
    C.use_se = False

    ########################################

    C.batch_size = 196
    C.niters_per_epoch = C.num_train_imgs // C.batch_size
    C.image_height = 32 # this size is after down_sampling
    C.image_width = 32

    C.save = "finetune"
    ########################################
    # TODO:
    # C.nepochs = 300
    C.nepochs = 600

    C.eval_epoch = 100

    C.lr_schedule = 'cosine'

    C.lr = 0.02
    # linear 
    C.decay_epoch = 100
    # exponential
    C.lr_decay = 0.97
    # multistep
    # C.milestones = [80, 120, 160]
    C.milestones = [200, 400]
    C.gamma = 0.1
    # cosine
    C.learning_rate_min = 0.001

    C.load_path = '/media/HardDisk1/shihh/NAS/ckpt/CIFAR10_AddShiftAdd_322'

    C.eval_only = False

    C.efficiency_metric = 'latency'
    C.hw_platform_path = 'fbnet/edgegpu/'

elif 'imagenet' in C.dataset:
    """Data Dir and Weight Dir"""
    # TODO:
    C.dataset_path = "/media/HardDisk1/cifar/CIFAR10"
   
    C.num_classes = 1000

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
    # TODO:
    C.pretrain = '/media/HardDisk1/shihh/NAS/ckpt/CIFAR10_AddShiftAdd_scratch_70_int16'
    C.search_space = 'AddShiftAdd'

    C.num_layer_list = [1, 4, 4, 4, 4, 4, 1]
    C.num_channel_list = [16, 24, 32, 64, 112, 184, 352]
    C.stride_list = [1, 2, 2, 2, 1, 2, 1]

    C.stem_channel = 16
    C.header_channel = 1504

    C.use_hswish = False
    C.use_se = False

    ########################################

    C.batch_size = 512
    C.niters_per_epoch = C.num_train_imgs // C.batch_size
    C.image_height = 224 # this size is after down_sampling
    C.image_width = 224

    C.save = "finetune"
    ########################################
    # TODO:
    # C.nepochs = 300
    C.nepochs = 200

    C.eval_epoch = 100

    C.lr_schedule = 'cosine'

    C.lr = 0.05
    # linear 
    C.decay_epoch = 100
    # exponential
    C.lr_decay = 0.97
    # multistep
    C.milestones = [80, 120, 160]
    C.gamma = 0.1
    # cosine
    C.learning_rate_min = 0.001

    C.load_path = '/media/HardDisk1/shihh/NAS/ckpt/CIFAR10_AddShiftAdd_322'

    C.eval_only = False

    C.efficiency_metric = 'latency'
    C.hw_platform_path = 'fbnet/edgegpu/'

else:
    print('Wrong dataset.')
    sys.exit()
