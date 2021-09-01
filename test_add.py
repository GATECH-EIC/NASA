from adder import adder
import torch
import torch.nn as nn
from torch.autograd import Variable
import ipdb

def Addlayer(in_planes, out_planes, kernel_size=1, stride=1, padding=1, groups=1, quantize=False, weight_bits=5, sparsity=0, quantize_v='sbm'):
    " 3x3 convolution with padding "
    return adder.Adder2D(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False,
                         quantize=quantize, weight_bits=weight_bits, sparsity=sparsity, quantize_v=quantize_v)

class AddNet(nn.Module):
    def __init__(self):
        super(AddNet, self).__init__()

        self.add = Addlayer(4, 4, kernel_size=3, stride=1, padding=1, groups=4)


    def forward(self, input):
        
        out = self.add(input)

        return out

# x = torch.rand(2,4,16,16,requires_grad=True).cuda()

x = Variable(torch.random(2,4,16,16), requires_grad=True).cuda()
model = AddNet()
y = model(x)
# print(y.size())
# loss = y-x
y.backward()

