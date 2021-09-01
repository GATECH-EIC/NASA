import os
import sys
import time
import glob
import numpy as np
import pickle
import torch
import logging
import argparse
import torch
import random


torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
random.seed(0)
torch.backends.cudnn.deterministic = True

# from network import ShuffleNetV2_OneShot
from model_search_ws import FBNet as Network
# from config_search import config

from tester import get_cand_err
# from flops import get_cand_flops

from torch.autograd import Variable
import collections
import sys
sys.setrecursionlimit(10000)
import argparse

import functools
print = functools.partial(print, flush=True)

choice = lambda x: x[np.random.randint(len(x))] if isinstance(
    x, tuple) else choice(tuple(x))


class EvolutionSearcher(object):

    def __init__(self, args):
        self.args = args

        self.max_epochs = args.max_epochs
        self.select_num = args.select_num
        self.population_num = args.population_num
        self.m_prob = args.m_prob
        self.crossover_num = args.crossover_num
        self.mutation_num = args.mutation_num
        # self.flops_limit = args.flops_limit

        # self.model = ShuffleNetV2_OneShot()
        self.model = Network(config=args)
        self.model = torch.nn.DataParallel(self.model).cuda()
        # supernet_state_dict = torch.load(
        #     '/media/HardDisk1/shihh/NAS/ckpt/CIFAR100_AddAll_noshiftadd_supernet/weights_pretrain_240.pth')['state_dict']
        # self.model.load_state_dict(supernet_state_dict)
        partial = torch.load(args.load_path)
        # partial = torch.load(pretrain + "/weights_pretrain_110.pth")
        state = self.model.state_dict()
        pretrained_dict = {k: v for k, v in partial.items() if k in state and state[k].size() == partial[k].size()}
        state.update(pretrained_dict)
        self.model.load_state_dict(state)
        self.log_dir = args.log_dir
        self.checkpoint_name = os.path.join(self.log_dir, 'checkpoint.pth.tar')

        self.memory = []
        self.vis_dict = {}
        self.keep_top_k = {self.select_num: [], self.population_num: []}
        self.epoch = 0
        self.candidates = []

        self.nr_layer = 22
        self.nr_state = args.nr_state

    def save_checkpoint(self):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        info = {}
        info['memory'] = self.memory
        info['candidates'] = self.candidates
        info['vis_dict'] = self.vis_dict
        info['keep_top_k'] = self.keep_top_k
        info['epoch'] = self.epoch
        torch.save(info, self.checkpoint_name)
        print('save checkpoint to', self.checkpoint_name)

    def load_checkpoint(self):
        if not os.path.exists(self.checkpoint_name):
            return False
        info = torch.load(self.checkpoint_name)
        self.memory = info['memory']
        self.candidates = info['candidates']
        self.vis_dict = info['vis_dict']
        self.keep_top_k = info['keep_top_k']
        self.epoch = info['epoch']

        print('load checkpoint from', self.checkpoint_name)
        return True

    def is_legal(self, cand):
        assert isinstance(cand, tuple) and len(cand) == self.nr_layer
        if cand not in self.vis_dict:
            self.vis_dict[cand] = {}
        info = self.vis_dict[cand]
        if 'visited' in info:
            return False

        # if 'flops' not in info:
        #     info['flops'] = get_cand_flops(cand)

        # print(cand, info['flops'])

        # if info['flops'] > self.flops_limit:
        #     print('flops limit exceed')
        #     return False

        info['err'] = get_cand_err(self.model, cand, self.args)

        info['visited'] = True

        return True

    def update_top_k(self, candidates, *, k, key, reverse=False):
        assert k in self.keep_top_k
        print('select ......')
        t = self.keep_top_k[k]
        t += candidates
        t.sort(key=key, reverse=reverse)
        self.keep_top_k[k] = t[:k]

    def stack_random_cand(self, random_func, *, batchsize=10):
        while True:
            cands = [random_func() for _ in range(batchsize)]
            for cand in cands:
                if cand not in self.vis_dict:
                    self.vis_dict[cand] = {}
                info = self.vis_dict[cand]
            for cand in cands:
                yield cand

    def get_random(self, num):
        print('random select ........')
        cand_iter = self.stack_random_cand(
            lambda: tuple(np.random.randint(self.nr_state) for i in range(self.nr_layer)))
        while len(self.candidates) < num:
            cand = next(cand_iter)
            if not self.is_legal(cand):
                continue
            self.candidates.append(cand)
            print('random {}/{}'.format(len(self.candidates), num))
        print('random_num = {}'.format(len(self.candidates)))

    def get_mutation(self, k, mutation_num, m_prob):
        assert k in self.keep_top_k
        print('mutation ......')
        res = []
        iter = 0
        max_iters = mutation_num * 10

        def random_func():
            cand = list(choice(self.keep_top_k[k]))
            for i in range(self.nr_layer):
                if np.random.random_sample() < m_prob:
                    cand[i] = np.random.randint(self.nr_state)
            return tuple(cand)

        cand_iter = self.stack_random_cand(random_func)
        while len(res) < mutation_num and max_iters > 0:
            max_iters -= 1
            cand = next(cand_iter)
            if not self.is_legal(cand):
                continue
            res.append(cand)
            print('mutation {}/{}'.format(len(res), mutation_num))

        print('mutation_num = {}'.format(len(res)))
        return res

    def get_crossover(self, k, crossover_num):
        assert k in self.keep_top_k
        print('crossover ......')
        res = []
        iter = 0
        max_iters = 10 * crossover_num

        def random_func():
            p1 = choice(self.keep_top_k[k])
            p2 = choice(self.keep_top_k[k])
            return tuple(choice([i, j]) for i, j in zip(p1, p2))
        cand_iter = self.stack_random_cand(random_func)
        while len(res) < crossover_num and max_iters > 0:
            max_iters -= 1
            cand = next(cand_iter)
            if not self.is_legal(cand):
                continue
            res.append(cand)
            print('crossover {}/{}'.format(len(res), crossover_num))

        print('crossover_num = {}'.format(len(res)))
        return res

    def search(self):
        print('population_num = {} select_num = {} mutation_num = {} crossover_num = {} random_num = {} max_epochs = {}'.format(
            self.population_num, self.select_num, self.mutation_num, self.crossover_num, self.population_num - self.mutation_num - self.crossover_num, self.max_epochs))

        # self.load_checkpoint()

        self.get_random(self.population_num)

        while self.epoch < self.max_epochs:
            print('epoch = {}'.format(self.epoch))

            self.memory.append([])
            for cand in self.candidates:
                self.memory[-1].append(cand)

            self.update_top_k(
                self.candidates, k=self.select_num, key=lambda x: self.vis_dict[x]['err'])
            self.update_top_k(
                self.candidates, k=self.population_num, key=lambda x: self.vis_dict[x]['err'])

            cands = sorted([cand for cand in self.vis_dict if 'err' in self.vis_dict[cand]], key=lambda cand: self.vis_dict[cand]['err'])[:1][0]
            
            self.model.module.show_arch(cands=cands)

            print('epoch = {} : top {} result'.format(
                self.epoch, len(self.keep_top_k[self.population_num])))
            for i, cand in enumerate(self.keep_top_k[self.population_num]):
                print('No.{} {} Top-1 err = {}'.format(
                    i + 1, cand, self.vis_dict[cand]['err']))
                ops = [i for i in cand]
                print(ops)

            mutation = self.get_mutation(
                self.select_num, self.mutation_num, self.m_prob)
            crossover = self.get_crossover(self.select_num, self.crossover_num)

            self.candidates = mutation + crossover

            self.get_random(self.population_num)

            self.epoch += 1

        self.save_checkpoint()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, default='Evolutionary2')
    parser.add_argument('--max-epochs', type=int, default=20)
    parser.add_argument('--select-num', type=int, default=10)
    parser.add_argument('--population-num', type=int, default=50)
    parser.add_argument('--m_prob', type=float, default=0.1)
    parser.add_argument('--crossover-num', type=int, default=25)
    parser.add_argument('--mutation-num', type=int, default=25)
    parser.add_argument('--max-train-iters', type=int, default=200)
    parser.add_argument('--max-test-iters', type=int, default=40)
    parser.add_argument('--train-batch-size', type=int, default=128)
    parser.add_argument('--test-batch-size', type=int, default=200)
    parser.add_argument('--nr_state', type=int, default=19)
    ####################################################
    parser.add_argument('--hard', type=bool, default=True,
                    help='sample function')
    parser.add_argument('--mode', type=str, default='proxy_hard',
                        help='the mode of sample function')
    parser.add_argument('--act_num', type=int, default=1,
                        help='the number of active path')
    parser.add_argument('--num_classes', type=int, default=100,
                        help='the class number of dataset')
    parser.add_argument('--num_layer_list', type=list, default=[1, 4, 4, 4, 4, 4, 1],
                        help='num_layer_list')
    parser.add_argument('--num_channel_list', type=list, default=[16, 24, 32, 64, 112, 184, 352],
                        help='num_channel_list')
    parser.add_argument('--stride_list', type=list, default=[1, 1, 2, 2, 1, 2, 1],
                        help='num_channel_list')
    parser.add_argument('--stem_channel', type=int, default=16,
                        help='num_channel_list')
    parser.add_argument('--header_channel', type=int, default=1504,
                        help='header_channel')
    parser.add_argument('--search_space', type=str, default='AddAll',
                        help='the search space')
    parser.add_argument('--dataset', type=str, default='cifar100',
                        help='dataset')
    parser.add_argument('--dataset_path', type=str, default='/media/HardDisk1/cifar/CIFAR100', help='the path of dataset')
    parser.add_argument('--load_path', type=str, default='/media/HardDisk1/cifar/CIFAR100', help='the path of checkpoint')
    parser.add_argument('--sample_func', type=str, default='gumbel_softmax',
                        help='number of workers')
    parser.add_argument('--gpu', type=str, default='0', 
                        help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--port", type=str, default="10001")
    parser.add_argument('--distributed', type=bool, default=False, 
                        help='whether to use distributed training')
    parser.add_argument("--ngpus_per_node", type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1,
                        help='number of nodes')
    parser.add_argument('--rank', type=int, default=None,
                        help='node rank')
    parser.add_argument('--dist_url', type=str, default=None,
                        help='url used to set up distributed training')
    args = parser.parse_args()

    gpu_ids = args.gpu.split(',')
    # print("gpu_ids",gpu_ids)
    args.gpu = []
    for gpu_id in gpu_ids:
        id = int(gpu_id)
        # print("id",id)
        args.gpu.append(id)
    gpu = args.gpu
    print("gpu",gpu)

    if args.dataset == 'cifar10':
        args.num_classes = 10
    elif args.dataset == 'cifar100':
        args.num_classes = 100
    else:
        args.num_classes = 100
        print('Dataset: imagenet !')
    
    print(args)

    t = time.time()

    searcher = EvolutionSearcher(args)

    searcher.search()

    print('total searching time = {:.2f} hours'.format(
        (time.time() - t) / 3600))

if __name__ == '__main__':
    try:
        main()
        os._exit(0)
    except:
        import traceback
        traceback.print_exc()
        time.sleep(1)
        os._exit(1)
