import torch
import os
import sys
# from imagenet_dataset import get_train_dataprovider, get_val_dataprovider
import tqdm
import torchvision.transforms as transforms
import torchvision.datasets as dset
from datasets import prepare_train_data, prepare_test_data, prepare_train_data_for_search, prepare_test_data_for_search

assert torch.cuda.is_available()

# train_dataprovider, val_dataprovider = None, None

class DataIterator(object):

    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iterator = enumerate(self.dataloader)

    def next(self):
        try:
            _, data = next(self.iterator)
        except Exception:
            self.iterator = enumerate(self.dataloader)
            _, data = next(self.iterator)
        return data[0], data[1]

def accuracy(output, target, topk=(1,)):
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


def no_grad_wrapper(func):
    def new_func(*args, **kwargs):
        with torch.no_grad():
            return func(*args, **kwargs)
    return new_func


@no_grad_wrapper
def get_cand_err(model, cand, args):
    # global train_dataprovider, val_dataprovider

    # if train_dataprovider is None:
    #     use_gpu = False
    #     train_dataprovider = get_train_dataprovider(
    #         args.train_batch_size, use_gpu=False, num_workers=8)
    #     val_dataprovider = get_val_dataprovider(
    #         args.test_batch_size, use_gpu=False, num_workers=8)
    # # data loader ###########################
    if 'cifar' in args.dataset:
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

        if args.dataset == 'cifar10':
            train_data = dset.CIFAR10(root=args.dataset_path, train=True, download=False, transform=transform_train)
            test_data = dset.CIFAR10(root=args.dataset_path, train=False, download=False, transform=transform_test)
        elif args.dataset == 'cifar100':
            train_data = dset.CIFAR100(root=args.dataset_path, train=True, download=False, transform=transform_train)
            # train_data.data = train_data.data[:32]
            test_data = dset.CIFAR100(root=args.dataset_path, train=False, download=False, transform=transform_test)
            # test_data.data = test_data.data[:32]
        else:
            print('Wrong dataset.')
            sys.exit()


    elif args.dataset == 'imagenet':
        train_data = prepare_train_data_for_search(dataset=args.dataset,
                                          datadir=args.dataset_path+'/train', num_class=args.num_classes)
        test_data = prepare_test_data_for_search(dataset=args.dataset,
                                        datadir=args.dataset_path+'/val', num_class=args.num_classes)

    elif args.dataset == 'tinyimagenet':
        print('Wrong dataset.')
        traindir = os.path.join(args.dataset_path, 'train')
        valdir = os.path.join(args.dataset_path, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])

        train_data = dset.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
        
        test_data = dset.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))
    else:
        print("Wrong dataset!")

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.train_batch_size, shuffle=True,
        num_workers=8, pin_memory=False)
    train_dataprovider = DataIterator(train_loader)

    val_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.test_batch_size, shuffle=False,
        num_workers=8, pin_memory=False
    )
    val_dataprovider = DataIterator(val_loader)

    max_train_iters = args.max_train_iters
    max_test_iters = args.max_test_iters

    print('clear bn statics....')
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.running_mean = torch.zeros_like(m.running_mean)
            m.running_var = torch.ones_like(m.running_var)
            # print("BN exists!!")

    print('train bn with training set (BN sanitize) ....')
    model.train()

    # for step in tqdm.tqdm(range(max_train_iters)):
    for step in range(max_train_iters):
        # print('train step: {} total: {}'.format(step,max_train_iters))
        data, target = train_dataprovider.next()
        # print('get data',data.shape)

        # target = target.type(torch.LongTensor)

        # data, target = data.to(device), target.to(device)
        data = data.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        output = model(input=data, cand=cand)

        del data, target, output

    top1 = 0
    top5 = 0
    total = 0

    print('starting test....')
    model.eval()
    prec1_list = []
    prec5_list = []
    # for step in tqdm.tqdm(range(max_test_iters)):
    # for step in tqdm.tqdm(range(max_test_iters)):
    for step in range(max_test_iters):
        # print('test step: {} total: {}'.format(step,max_test_iters))
        data, target = val_dataprovider.next()
        batchsize = data.shape[0]
        # print('get data',data.shape)
        # target = target.type(torch.LongTensor)
        # data, target = data.to(device), target.to(device)
        data = data.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        logits = model(input=data, cand=cand)

        prec1, prec5 = accuracy(logits, target, topk=(1, 5))

        # print(prec1.item(),prec5.item())
        prec1_list.append(prec1)
        prec5_list.append(prec5)
        # top1 += prec1.item() * batchsize
        # top5 += prec5.item() * batchsize
        # total += batchsize

        del data, target, logits, prec1, prec5

    # top1, top5 = top1 / total, top5 / total

    # top1, top5 = 1 - top1 / 100, 1 - top5 / 100
    top1 = sum(prec1_list)/len(prec1_list)
    top5 = sum(prec5_list)/len(prec5_list)

    print('top1:%.3f top5:%.3f' %(top1, top5))

    return top1, top5


def main():
    pass
