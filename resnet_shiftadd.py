# from models import adder
from adder import adder
import torch.nn as nn
from deepshift import modules
from deepshift import modules_q
from deepshift import utils as utils


__all__ = ['resnet20_shiftadd']

def shift_conv(in_planes, out_planes, kernel_size=1, stride=1, padding=1, bias=False, freeze_sign = False, use_kernel=False, use_cuda=True,
    rounding='deterministic', weight_bits=5, sign_threshold_ps=None, quant_bits=16):
    
    shift_conv2d = modules.Conv2dShift(in_channels=in_planes, out_channels=out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, freeze_sign=freeze_sign, use_kernel=use_kernel, use_cuda=use_cuda,
                                    rounding=rounding, weight_bits=weight_bits, threshold=sign_threshold_ps, quant_bits=quant_bits)


    # model._modules[name] = shift_conv2d
    # conversion_count += 1

    return shift_conv2d


def conv3x3(in_planes, out_planes, stride=1, quantize=False, weight_bits=8, quantize_v='sbm'):
    " 3x3 convolution with padding "
    shift = shift_conv(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
    add = adder.Adder2D(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False, quantize=quantize, weight_bits=weight_bits, quantize_v=quantize_v)
    return nn.Sequential(shift, add)
    # return adder.Adder2D(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False, quantize=quantize, weight_bits=weight_bits, quantize_v=quantize_v)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, quantize=False, weight_bits=8, quantize_v='sbm'):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride=stride, quantize=quantize, weight_bits=weight_bits, quantize_v=quantize_v)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, quantize=quantize, weight_bits=weight_bits, quantize_v=quantize_v)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10, quantize=False, weight_bits=8, quantize_v='sbm'):
        super(ResNet, self).__init__()
        self.inplanes = 16
        self.quantize = quantize
        self.weight_bits = weight_bits
        self.quantize_v = quantize_v
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        # use conv as fc layer (addernet)
        self.fc = nn.Conv2d(64 * block.expansion, num_classes, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_classes)

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), # shift
                adder.Adder2D(planes * block.expansion, planes * block.expansion, kernel_size=1, stride=1, bias=False,
                              quantize=self.quantize, weight_bits=self.weight_bits, quantize_v=self.quantize_v), # add
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(inplanes=self.inplanes, planes=planes, stride=stride, downsample=downsample,
                            quantize=self.quantize, weight_bits=self.weight_bits, quantize_v=self.quantize_v))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes=self.inplanes, planes=planes,
                                quantize=self.quantize, weight_bits=self.weight_bits, quantize_v=self.quantize_v))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = self.fc(x)
        x = self.bn2(x)
        return x.view(x.size(0), -1)


class ResNet_vis(nn.Module):

    def __init__(self, block, layers, num_classes=10):
        super(ResNet_vis, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        # use linear as fc layer
        self.linear_1 = nn.Linear(64 * block.expansion, 2)
        self.linear_2 = nn.Linear(2, num_classes, bias=False)

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), # shift
                adder.Adder2D(planes * block.expansion, planes * block.expansion, kernel_size=1, stride=1, bias=False), # add
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(inplanes=self.inplanes, planes=planes, stride=stride, downsample=downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes=self.inplanes, planes=planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        feat = self.linear_1(x)
        x = self.linear_2(feat)
        return feat, x.view(x.size(0), -1)


def resnet20_shiftadd(num_classes=10, **kwargs):
    return ResNet(BasicBlock, [3, 3, 3], num_classes=num_classes, **kwargs)
