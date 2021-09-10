from __future__ import division
import os
import sys
import time
import glob
import logging
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as dset
from torch.autograd import Variable

import torchvision

import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed

from tensorboardX import SummaryWriter

from config_train import config

from datasets import prepare_train_data, prepare_test_data

import numpy as np
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from PIL import Image

from config_train import config
import genotypes

from model_search import FBNet as Network
from model_infer_distill import FBNet_Infer

from lr import LambdaLR

from thop import profile
# from thop.count_hooks import count_convNd
from resnet20_add import resnet20_add

import argparse



parser = argparse.ArgumentParser(description='DNA')
parser.add_argument('--dataset', type=str, default=None,
                    help='type of dataset')
parser.add_argument('--dataset_path', type=str, default=None,
                    help='path to dataset')
parser.add_argument('--pretrain', type=str, default=None,
                    help='path to searched arch')
parser.add_argument('--search_space', type=str, default=None,
                    help='choice of search_space')
parser.add_argument('--load_path', type=str, default=None,
                    help='path to trained models')
parser.add_argument('--batch_size', type=int, default=None,
                    help='batch size')
parser.add_argument('--header_channel', type=int, default=1504,
                    help='header_channel')
parser.add_argument('--lr', type=float, default=None,
                    help='the learning rate')
parser.add_argument('--lr_schedule', type=str, default=None,
                    help='the learning rate schedule')
parser.add_argument('--weight_decay', type=float, default=None,
                    help='the weight_decay')
parser.add_argument('--nepochs', type=int, default=None,
                    help='training epochs')
parser.add_argument('--transfer_epoch', type=int, default=None,
                    help='the epoch to transfer the weight of teacher to the student')
parser.add_argument('--load_epoch', type=int, default=None,
                    help='which epoch to load')
parser.add_argument('--gpu', type=str, default='0', 
                    help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
# parser.add_argument('--gpu', nargs='+', type=int, default=None,
#                     help='specify gpus')
# distributed parallel
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument("--port", type=str, default="10001")
parser.add_argument("--spos", type=bool, default=False,
                    help='whether to use spos')
parser.add_argument('--distributed', type=bool, default=False, 
                    help='whether to use distributed training')
parser.add_argument('--distillation', type=bool, default=False, 
                    help='whether to use knowledge distillation')
parser.add_argument("--ngpus_per_node", type=int, default=0)
parser.add_argument('--num_workers', type=int, default=None,
                    help='number of workers per gpu')
parser.add_argument('--world_size', type=int, default=None,
                    help='number of nodes')
parser.add_argument('--rank', type=int, default=None,
                    help='node rank')
parser.add_argument('--dist_url', type=str, default=None,
                    help='url used to set up distributed training')
args = parser.parse_args()


best_acc = 0
best_epoch = 0
distil_loss = 5e-3
cudnn.benchmark = True

def main():
    if args.dataset is not None:
        config.dataset = args.dataset
    if args.dataset_path is not None:
        config.dataset_path = args.dataset_path
    if args.nepochs is not None:
        config.nepochs = args.nepochs
    if args.load_epoch is not None:
        config.load_epoch = args.load_epoch
    if args.transfer_epoch is not None:
        config.transfer_epoch = args.load_epoch
    if args.header_channel is not None:
        config.header_channel = args.header_channel
    if args.spos is not None:
        config.spos = args.spos
    if args.load_path is not None:
        config.load_path = args.load_path
    if args.pretrain is not None:
        config.pretrain = args.pretrain
    if args.distillation is not None:
        config.distillation = args.distillation
    if args.search_space is not None:
        config.search_space = args.search_space
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.lr is not None:
        config.lr = args.lr
    if args.lr_schedule is not None:
        config.lr_schedule = args.lr_schedule
    if args.weight_decay is not None:
        config.weight_decay = args.weight_decay
    if args.num_workers is not None:
        config.num_workers = args.num_workers
    if args.world_size is not None:
        config.world_size = args.world_size
    if args.world_size is not None:
        config.rank = args.rank
    if args.dist_url is not None:
        config.dist_url = args.dist_url
    if args.gpu is not None:
        config.gpu = args.gpu
    if args.port is not None:
        config.port = args.port
    config.distributed = args.distributed
    if args.local_rank is not None:
        config.local_rank = args.local_rank
    if args.ngpus_per_node is not None:
        ngpus_per_node = args.ngpus_per_node

    gpu_ids = config.gpu.split(',')
    # print("gpu_ids",gpu_ids)
    config.gpu = []
    for gpu_id in gpu_ids:
        id = int(gpu_id)
        # print("id",id)
        config.gpu.append(id)
    gpu = config.gpu
    print("gpu",gpu)
    if config.dataset == 'cifar10':
        config.num_classes = 10
    elif config.dataset == 'cifar100':
        config.num_classes = 100
    else:
        print('Dataset: imagenet !')
        # sys.exit()

    config.distributed = config.world_size > 1 or config.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    config.ngpus_per_node = ngpus_per_node
    config.num_workers = config.num_workers * ngpus_per_node

    if config.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        config.world_size = ngpus_per_node * config.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, config))
    else:
        # Simply call main_worker function
        config.world_size = ngpus_per_node * config.world_size
        main_worker(config.gpu, ngpus_per_node, config)


def main_worker(gpu, ngpus_per_node, config):
    global best_acc
    global best_epoch
    global distil_loss

    config.gpu = gpu
    pretrain = config.pretrain

    if config.gpu is not None:
        print("Use GPU: {} for training".format(config.gpu))


    if config.distributed:
        if config.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            config.rank = config.rank * ngpus_per_node + gpu
        # dist.init_process_group(backend=config.dist_backend, init_method=config.dist_url,
        #                         world_size=config.world_size, rank=config.rank)
        # print("Rank: {}".format(config.rank))
        os.environ['MASTER_PORT'] = config.port
        dist.init_process_group(backend="nccl")

    if not (config.multiprocessing_distributed or config.distributed) or (config.multiprocessing_distributed and config.rank % ngpus_per_node == 0) or (config.distributed and dist.get_rank() == 0):
        if type(pretrain) == str:
            config.save = pretrain
        else:
            config.save = 'ckpt/{}-{}'.format(config.save, time.strftime("%Y%m%d-%H%M%S"))

        logger = SummaryWriter(config.save)
        log_format = '%(asctime)s %(message)s'
        logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
        fh = logging.FileHandler(os.path.join(config.save, 'log.txt'))
        fh.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(fh)

        logging.info("args = %s", str(config))
    else:
        logger = None


    # Model #######################################
    # TODO:
    if (config.spos==True):
        info = torch.load(os.path.join(config.load_path,'checkpoint.pth.tar'))['vis_dict']
        cands = sorted([cand for cand in info if 'err' in info[cand]],
                   key=lambda cand: info[cand]['err'])[:1][0]
        model = FBNet_Infer(alpha=None, config=config, cand=cands)
        Epoch = 0
    else:
        state = torch.load(os.path.join(config.load_path, 'arch_%s.pth' %config.load_epoch))
        alpha = state['alpha']
        # alpha = torch.tensor([
    #     [0.0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],
    #     [0.0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    #     [0.0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],
    #     [0.0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    #     [0.0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    #     [0.0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    #     [0.0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],
    #     [0.0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
    #     [0.0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],
    #     [0.0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    #     [0.0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
    #     [0.0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],
    #     [0.0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    #     [0.0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    #     [0.0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
    #     [0.0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],
    #     [0.0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],
    #     [0.0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
    #     [0.0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],
    #     [0.0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    #     [0.0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
    #     [0.0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
    # ])
        
        # print(alpha)
        Epoch = state['epoch']
        print(Epoch)
        # print('ok!!!!!')
        model = FBNet_Infer(alpha=alpha, config=config, cand=None)
        # print('ok!!!!!')
        flops = model.forward_flops((3, config.image_height, config.image_width))
        print('FLOPs: ', flops)
        # model = resnet20_add(num_classes=100, quantize=False, weight_bits=8, quantize_v='sbm')
        if config.distillation == True:
            print('Distillation !!!!!!!!!!!!!!')
            model_teacher = FBNet_Infer(alpha=alpha, config=config, flag=True, cand=None)
            # print(model_teacher)

    print('config.gpu:', config.gpu)
    if config.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        # if config.gpu is not None:
        if len(config.gpu) > 1:
            # torch.cuda.set_device(config.gpu)
            # model.cuda(config.gpu)
            model.cuda()
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            # config.batch_size = int(config.batch_size / ngpus_per_node)
            config.num_workers = int((config.num_workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=config.gpu, find_unused_parameters=True)
            if config.distillation == True:
                model_teacher.cuda()
                model_teacher = torch.nn.parallel.DistributedDataParallel(model_teacher, device_ids=config.gpu, find_unused_parameters=True)
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
            if config.distillation == True:
                model_teacher.cuda()
                model_teacher = torch.nn.parallel.DistributedDataParallel(model_teacher, find_unused_parameters=True)
    else:
        model = torch.nn.DataParallel(model).cuda()
        if config.distillation == True:
            model_teacher = torch.nn.DataParallel(model_teacher).cuda()


    # model_other_params = []
    # model_add_params = []

    # for name, param in model.named_parameters():
    #     if(name.endswith(".shift")):
    #         model_add_params.append(param)
    #     else:
    #         model_other_params.append(param)
    
    # params_dict = [
    # {"params": model_other_params},
    # {"params": model_add_params, 'lr': config.lr_add if config.lr_add is not None else config.lr, 'weight_decay': 0},
    # ]
    # TODO:
    # params_dict = [
    # {"params": model.parameters()},
    # {"params": list(model.module._var_params.values())},
    # ]

    # params_dict_teacher = [
    # {"params": model_teacher.parameters()},
    # {"params": list(model_teacher.module._var_params.values())},
    # ]
    para = model.parameters()
    if config.opt == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.lr,
            betas=config.betas)
        if config.distillation == True:
            optimizer_teacher = torch.optim.Adam(
            model_teacher.parameters(),
            lr=config.lr,
            betas=config.betas)
    elif config.opt == 'Sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay)
        if config.distillation == True:
            optimizer_teacher = torch.optim.SGD(
            model_teacher.parameters(),
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay)
    else:
        print("Wrong Optimizer Type.")
        sys.exit()

    # print(model.parameters())

    # lr policy ##############################
    # total_iteration = config.nepochs * config.niters_per_epoch
    
    if config.lr_schedule == 'linear':
        lr_policy = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=LambdaLR(config.nepochs, 0, config.decay_epoch).step)
        if config.distillation == True:
            lr_policy_teacher = torch.optim.lr_scheduler.LambdaLR(optimizer_teacher, lr_lambda=LambdaLR(config.nepochs, 0, config.decay_epoch).step)
    elif config.lr_schedule == 'exponential':
        lr_policy = torch.optim.lr_scheduler.ExponentialLR(optimizer, config.lr_decay)
        if config.distillation == True:
            lr_policy_teacher = torch.optim.lr_scheduler.ExponentialLR(optimizer_teacher, config.lr_decay)
    elif config.lr_schedule == 'multistep':
        lr_policy = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.milestones, gamma=config.gamma)
        if config.distillation == True:
            lr_policy_teacher = torch.optim.lr_scheduler.MultiStepLR(optimizer_teacher, milestones=config.milestones, gamma=config.gamma)
    elif config.lr_schedule == 'cosine':
        lr_policy = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(config.nepochs), eta_min=config.learning_rate_min)
        if config.distillation == True:
            lr_policy_teacher = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_teacher, float(config.nepochs), eta_min=config.learning_rate_min)
    else:
        print("Wrong Learning Rate Schedule Type.")
        sys.exit()

    cudnn.benchmark = True


    # if use multi machines, the pretrained weight and arch need to be duplicated on all the machines
    # TODO:
    # if type(pretrain) == str and os.path.exists(pretrain + "/weights_best_%d.pth" %(config.load_epoch)):
    #     pretrained_model = torch.load(pretrain + "/weights_best_%d.pth" %(config.load_epoch))
    if type(pretrain) == str and os.path.exists("/media/shared-corpus/shh/NAS/adder/CIFAR100_AddAdd_0.1_scratch_wd1e-4_distill/weights_best_240.pth"):
        pretrained_model = torch.load("/media/shared-corpus/shh/NAS/adder/CIFAR100_AddAdd_0.1_scratch_wd1e-4_distill/weights_best_240.pth")
        partial = pretrained_model['state_dict']
        # print("ckpt:",partial)
        state = model.state_dict()
        pretrained_dict = {k: v for k, v in partial.items() if k in state and state[k].size() == partial[k].size()}
        state.update(pretrained_dict)
        model.load_state_dict(state)
        
        # pretrain_arch = torch.load(pretrain + "/arch_40.pth")
        # model.module.alpha.data = pretrain_arch['alpha'].data

        # optimizer.load_state_dict(pretrained_model['optimizer'])
        # lr_policy.load_state_dict(pretrained_model['lr_scheduler'])
        # start_epoch = pretrained_model['epoch'] + 1
        
        start_epoch = 0
        # architect.optimizer.load_state_dict(pretrain_arch['arch_optimizer'])

        # best_acc = pretrained_model['best_acc']
        # best_epoch = pretrained_model['best_epoch']

        # print('Resume from Epoch %d. Load pretrained weight.' % start_epoch)
        print('Resume from Epoch. Load pretrained weight.')

    else:
        start_epoch = 0
        print('No checkpoint. Train from scratch.')

    if config.distillation == True:
        # if type(pretrain) == str and os.path.exists(pretrain + "/teacher_weights_corres_%d.pth" %(config.load_epoch)):
        #     pretrained_model = torch.load(pretrain + "/teacher_weights_corres_%d.pth" %(config.load_epoch))
        if type(pretrain) == str and os.path.exists("/media/shared-corpus/shh/NAS/adder/CIFAR100_AddAdd_0.1_scratch_wd1e-4_distill/weights_best_240.pth"):
            pretrained_model = torch.load("/media/shared-corpus/shh/NAS/adder/CIFAR100_AddAdd_0.1_scratch_wd1e-4_distill/weights_best_240.pth")
            partial = pretrained_model['state_dict']
            # print("ckpt:",partial)
            state = model_teacher.state_dict()
            pretrained_dict = {k: v for k, v in partial.items() if k in state and state[k].size() == partial[k].size()}
            state.update(pretrained_dict)
            model_teacher.load_state_dict(state)
            
            # pretrain_arch = torch.load(pretrain + "/arch_40.pth")
            # model.module.alpha.data = pretrain_arch['alpha'].data

            # optimizer_teacher.load_state_dict(pretrained_model['optimizer'])
            # lr_policy_teacher.load_state_dict(pretrained_model['lr_scheduler'])
            # start_epoch = pretrained_model['epoch'] + 1
            # architect.optimizer.load_state_dict(pretrain_arch['arch_optimizer'])

            best_acc_teacher = pretrained_model['best_acc']
            best_epoch_teacher = pretrained_model['best_epoch']

            print('Resumed teacher model. Load pretrained weight.')

        else:
            # start_epoch = 0
            print('No checkpoint. Train from scratch.')


    # data loader ############################
    if 'cifar' in config.dataset:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        if config.dataset == 'cifar10':
            train_data = dset.CIFAR10(root=config.dataset_path, train=True, download=False, transform=transform_train)
            test_data = dset.CIFAR10(root=config.dataset_path, train=False, download=False, transform=transform_test)
        elif config.dataset == 'cifar100':
            train_data = dset.CIFAR100(root=config.dataset_path, train=True, download=False, transform=transform_train)
            test_data = dset.CIFAR100(root=config.dataset_path, train=False, download=False, transform=transform_test)
        else:
            print('Wrong dataset.')
            sys.exit()


    elif config.dataset == 'imagenet':
        train_data = prepare_train_data(dataset=config.dataset,
                                          datadir=config.dataset_path+'/train')
        test_data = prepare_test_data(dataset=config.dataset,
                                        datadir=config.dataset_path+'/val')

    else:
        print('Wrong dataset.')
        sys.exit()

    if config.distributed:
        # train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_data)

    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=config.batch_size, shuffle=(train_sampler is None),
        pin_memory=True, num_workers=config.num_workers, sampler=train_sampler)

    test_loader = torch.utils.data.DataLoader(test_data,
                                          batch_size=config.batch_size,
                                          shuffle=False,
                                          pin_memory=True,
                                          num_workers=config.num_workers)

    if config.eval_only:
        if not (config.multiprocessing_distributed or config.distributed) or (config.multiprocessing_distributed and config.rank % ngpus_per_node == 0) or (config.distributed and dist.get_rank() == 0):
            logging.info('Eval: acc = %f', infer(0, model, test_loader, logger))
        sys.exit(0)

    # tbar = tqdm(range(config.nepochs), ncols=80)
    best_acc_teacher = 0
    best_epoch_teacher = 0
    for epoch in range(start_epoch, config.nepochs):
        if not (config.multiprocessing_distributed or config.distributed) or (config.multiprocessing_distributed and config.rank % ngpus_per_node == 0) or (config.distributed and dist.get_rank() == 0):
            # tbar.set_description("[Epoch %d/%d][train...]" % (epoch + 1, config.nepochs))
            logging.info("[Epoch %d/%d] lr=%f" % (epoch + 1, config.nepochs, optimizer.param_groups[0]['lr']))

        # if config.distributed:
        #     train_sampler.set_epoch(epoch)

        if config.distillation == True:
            train(train_loader, model, optimizer, lr_policy, logger, epoch, config, model_teacher, optimizer_teacher)
            torch.cuda.empty_cache()
            lr_policy.step()
            lr_policy_teacher.step()
            # print("sita: ",model.module._var_params.values())
        else:
            train(train_loader, model, optimizer, lr_policy, logger, epoch, config)
            torch.cuda.empty_cache()
            lr_policy.step()

       
        eval_epoch = config.eval_epoch

        #validation
        # if (epoch+1) % eval_epoch == 0:
        with torch.no_grad():

            if config.distillation == True:
                acc_teacher = infer(epoch, model_teacher, test_loader, logger)
            
            acc = infer(epoch, model, test_loader, logger)

            if config.distributed:
                acc = reduce_tensor(acc, config.world_size)
                if config.distillation == True:
                    acc_teacher = reduce_tensor(acc_teacher, config.world_size)

        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch
            state = {}
            state['state_dict'] = model.state_dict()
            state['optimizer'] = optimizer.state_dict()
            state['lr_scheduler'] = lr_policy.state_dict()
            state['epoch'] = epoch 
            state['acc'] = acc
            state['best_acc'] = best_acc
            state['best_epoch'] = best_epoch
            torch.save(state, os.path.join(config.save, 'weights_best_%d.pth' %(Epoch+1)))

            if config.distillation == True:
                state_teacher = {}
                state_teacher['state_dict'] = model_teacher.state_dict()
                state_teacher['optimizer'] = optimizer_teacher.state_dict()
                state_teacher['lr_scheduler'] = lr_policy_teacher.state_dict()
                state_teacher['best_acc'] = best_acc_teacher
                state_teacher['best_epoch'] = best_epoch_teacher
                torch.save(state_teacher, os.path.join(config.save, 'teacher_weights_corres_%d.pth' %(Epoch+1)))

        if config.distillation == True:
            if acc_teacher > best_acc_teacher:
                best_acc_teacher = acc_teacher
                best_epoch_teacher = epoch

        if not (config.multiprocessing_distributed or config.distributed) or (config.multiprocessing_distributed and config.rank % ngpus_per_node == 0) or (config.distributed and dist.get_rank() == 0):
            logger.add_scalar('acc/val', acc, epoch)
            logging.info("Epoch:%d Acc:%.3f Best Acc:%.3f Best Epoch:%d" % (epoch, acc, best_acc, best_epoch))
            if config.distillation == True:
                logging.info("Teacher model: Epoch:%d Acc:%.3f Best Acc:%.3f Best Epoch:%d" % (epoch, acc_teacher, best_acc_teacher, best_epoch_teacher))

        # if (epoch+1) == 200 or (epoch+1) == 400:
        #     state = {}
        #     state['state_dict'] = model.state_dict()
        #     state['optimizer'] = optimizer.state_dict()
        #     state['lr_scheduler'] = lr_policy.state_dict()
        #     state['epoch'] = epoch 
        #     state['acc'] = acc
        #     state['best_acc'] = best_acc
        #     state['best_epoch'] = best_epoch
        #     torch.save(state, os.path.join(config.save, 'weights_%d.pth'%(epoch+1)))
        


    if not (config.multiprocessing_distributed or config.distributed) or (config.multiprocessing_distributed and config.rank % ngpus_per_node == 0) or (config.distributed and dist.get_rank() == 0):
        state = {}
        state['state_dict'] = model.state_dict()
        state['optimizer'] = optimizer.state_dict()
        state['lr_scheduler'] = lr_policy.state_dict()
        state['epoch'] = epoch 
        state['acc'] = acc
        state['best_acc'] = best_acc
        state['best_epoch'] = best_epoch
        torch.save(state, os.path.join(config.save, 'weights_latest_%d.pth' %(Epoch+1)))
 
        if config.distillation == True: 
            state_teacher = {}
            state_teacher['state_dict'] = model_teacher.state_dict()
            state_teacher['optimizer'] = optimizer_teacher.state_dict()
            state_teacher['lr_scheduler'] = lr_policy_teacher.state_dict()
            # state_teacher['epoch'] = epoch 
            # state_teacher['acc'] = acc
            state_teacher['best_acc'] = best_acc_teacher
            state_teacher['best_epoch'] = best_epoch_teacher
            torch.save(state_teacher, os.path.join(config.save, 'teacher_weights_latest_%d.pth' %(Epoch+1)))


def train(train_loader, model, optimizer, lr_policy, logger, epoch, config, model_tearcher=None, optimizer_teacher=None):
    global distil_loss
    model.train()
    if config.distillation == True:
        model_tearcher.train()
        T=3
        distil_loss = 0.05
        mse_loss = 0.05

    MSE = nn.MSELoss()
    for step, (input, target) in enumerate(train_loader):
        optimizer.zero_grad()
        if config.distillation == True:
            optimizer_teacher.zero_grad()

        start_time = time.time()

        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        data_time = time.time() - start_time

        out_student, middel_student = model(input)
        # out_student = model(input)
        if config.distillation == True:
            out_teacher, middle_teacher = model_tearcher(input, flag=True)
            loss_teacher = model.module._criterion(out_teacher, target)
            loss_teacher.backward()
            nn.utils.clip_grad_norm_(model_tearcher.parameters(), config.grad_clip)
            optimizer_teacher.step()
            optimizer_teacher.zero_grad()
            # softmax_func=nn.Softmax(dim=1)
            # out_teacher=softmax_func(out_teacher.data)
            # out_teacher=torch.clamp(out_teacher, 1e-4, 1, out=None)
            # distillation_loss = model.module._criterion(out_student, out_teacher)
            with torch.no_grad():
                out_teacher_2 = out_teacher.clone().detach()
            outputs_S = F.log_softmax(out_student/T, dim=1)
            outputs_T = F.softmax(out_teacher_2/T, dim=1)
            distillation_loss = T*T*nn.functional.kl_div(outputs_S,outputs_T)
            # print('kl loss ', distillation_loss)
            # distillation_loss = 0
            with torch.no_grad():
                middle_teacher_2 = []
                # print(len(middle_teacher))
                for i in range(22):
                    # print(i)
                    middle_teacher_2.append(middle_teacher[i].clone().detach())
            middel_loss = 0 
            for i in range(22):
                distillation_loss += mse_loss*MSE(middel_student[i], middle_teacher_2[i])
                # middel_loss += MSE(middel_student[i], middle_teacher_2[i])
                # print('MSE loss ',distillation_loss)
            distillation_loss = distil_loss * distillation_loss
            # distillation_loss.backward(retain_graph=True)

        loss = model.module._criterion(out_student, target)
        # loss = F.cross_entropy(out_student, target)
        if config.distillation == True:
            # loss = loss + distillation_loss + mse_loss*middel_loss
            loss += distillation_loss
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        optimizer.zero_grad()

        total_time = time.time() - start_time

        if step % 20 == 0:
            if not config.multiprocessing_distributed or (config.multiprocessing_distributed and config.rank % config.ngpus_per_node == 0):
                logging.info("[Epoch %d/%d][Step %d/%d] Loss=%.3f Time=%.3f Data Time=%.3f" % (epoch + 1, config.nepochs, step + 1, len(train_loader), loss.item(), total_time, data_time))
                logger.add_scalar('loss/train', loss, epoch*len(train_loader)+step)

    torch.cuda.empty_cache()
    del loss
    if config.distillation == True:
        del distillation_loss


def infer(epoch, model, test_loader, logger):
    model.eval()
    prec1_list = []
    with torch.no_grad():
        for i, (input, target) in enumerate(test_loader):
            input_var = Variable(input).cuda()
            target_var = Variable(target).cuda()

            output, middle = model(input_var)
            # output = model(input_var)
            prec1, = accuracy(output.data, target_var, topk=(1,))
            prec1_list.append(prec1)

        acc = sum(prec1_list)/len(prec1_list)

    return acc


def reduce_tensor(rt, n):
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= n
    return rt


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def save(model, model_path):
    torch.save(model.state_dict(), model_path)


if __name__ == '__main__':
    main() 
