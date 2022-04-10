import torch as tt
from torch import nn
from torch.utils.cpp_extension import load
import torch.nn.functional as F

# adder_cuda = load(
#     'adder_cuda', ['adder_cuda.cpp', 'adder_cuda_kernel.cu'], verbose=True)

from adder.adder import Adder2D
from adder.add2 import Adder2D_2
from adder.adder_slow import adder2d, adder2d_function

# help(adder_cuda)``1        1`

def check_forward():
    batch_size = 1
    in_channels = 64
    out_channels = 64
    # in_channels = 1
    # out_channels = 1
    in_size = 256
    # in_size = 3
    kernel_size = 3
    padding = 1
    groups = in_channels
    stride = 1
    out_size = (in_size + 2 * padding - kernel_size) // stride + 1
    # print(out_size)

    input  = tt.randn(batch_size, in_channels, in_size, in_size).cuda()
    print(input.size())
    weight = tt.randn(out_channels, 1, kernel_size, kernel_size).cuda()
    bias   = tt.randn(out_channels).cuda()
    
    out_ref = tt.randn(batch_size, out_channels, out_size, out_size).cuda()

    for i in range (in_channels):
        adder_ref = adder2d(1,
                        1,
                        kernel_size,
                        stride,
                        padding,
                        bias = True).cuda()
        adder_ref.adder.data.copy_(weight[i,:,:,:])
        adder_ref.b.data.copy_(bias[i])
        # print(input[:,i,:,:].size())
        out_ref[:,i,:,:] = adder_ref(input[:,i,:,:].view(batch_size, 1, in_size, in_size))

    adder = Adder2D(in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding,
                    groups,
                    bias = True,
                    eta = 0.2).cuda()
    adder.adder.data.copy_(weight)
    adder.b.data.copy_(bias)

    output = adder(input)
    # time_e = time.time()
    # print("hand_written_conv: {:.4f}us".format((time_e - time_b) * 1e6))

    # time_b = time.time()
    # out_ref = F.conv2d(input, weight, bias, padding=padding)
    # out_ref = adder2d_function(input, weight, stride, padding)
    # out_ref = adder_ref(input)
    # time_e = time.time()
    # print("builtin_conv: {:.4f}us".format((time_e - time_b) * 1e6))

    print("Forward: max error: {:.3e}".format(float((out_ref - output).abs().max())))


def check_grad_in():
    batch_size = 1
    in_channels = 64
    out_channels = 64
    in_size = 128
    kernel_size = 3
    padding = 1
    stride = 1
    groups = in_channels
    # batch_size = 1
    # in_channels = 1
    # out_channels = 1
    # in_size = 2
    # kernel_size = 2
    # padding = 0
    # stride = 1
    out_size = (in_size + 2 * padding - kernel_size) // stride + 1
    print(out_size)

    input = tt.randn(batch_size, in_channels, in_size, in_size).cuda()
    grad_input = tt.randn(batch_size, in_channels, in_size, in_size).cuda()
    input.requires_grad = True
    weight = tt.randn(out_channels, 1, kernel_size, kernel_size).cuda()
    weight.requires_grad = True
    bias = tt.randn(out_channels).cuda()
    grad_output = tt.randn(batch_size, out_channels, out_size, out_size).cuda()
    out_ref = tt.randn(batch_size, out_channels, out_size, out_size).cuda()

    for i in range (in_channels):
        adder_ref = adder2d(1,
                        1,
                        kernel_size,
                        stride,
                        padding,
                        bias = True).cuda()
        adder_ref.adder.data.copy_(weight[i,:,:,:])
        adder_ref.b.data.copy_(bias[i])
        # print(input[:,i,:,:].size())
        out_ref[:,i,:,:] = adder_ref(input[:,i,:,:].view(batch_size, 1, in_size, in_size))
        out_ref[:,i,:,:].backward(grad_output[:,i,:,:],retain_graph=True)
        grad_input[:,i,:,:] = input.grad[:,i,:,:].clone()
        input.grad.zero_()


    adder = Adder2D(in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding,
                    groups,
                    bias = True,
                    eta = 0.2).cuda()
    adder.adder.data.copy_(weight)
    adder.b.data.copy_(bias)

    output = adder(input)
    output.backward(grad_output)
    grad_ref = input.grad.clone()


    print("Grad input: max error: {:.3e}".format(((grad_input - grad_ref)).abs().max()))
    # print("grad_input",grad_input)


def check_grad_weight():
    batch_size = 1
    in_channels = 64
    out_channels = 64
    in_size = 128
    kernel_size = 1
    padding = 0
    stride = 1
    groups = in_channels
    out_size = (in_size + 2 * padding - kernel_size) // stride + 1
    print(out_size)

    input = tt.randn(batch_size, in_channels, in_size, in_size).cuda()
    # print(input)
    input.requires_grad = True
    weight = tt.randn(out_channels, 1, kernel_size, kernel_size).cuda()
    weight.requires_grad = True
    bias = tt.randn(out_channels).cuda()
    grad_output = tt.randn(batch_size, out_channels, out_size, out_size).cuda()
    grad_clone = tt.randn(out_channels, 1, kernel_size, kernel_size).cuda()
    out_ref = tt.randn(batch_size, out_channels, out_size, out_size).cuda()

    # print(input)
    # print(weight)
    # print(grad_output)

    for i in range (in_channels):
        adder_ref = adder2d(1,
                        1,
                        kernel_size,
                        stride,
                        padding,
                        bias = False).cuda()
        adder_ref.adder.data.copy_(weight[i,:,:,:])
        # adder_ref.b.data.copy_(bias[i])
        # print(input[:,i,:,:].size())
        output = adder_ref(input[:,i,:,:].view(batch_size, 1, in_size, in_size))
        output.backward(grad_output[:,i,:,:].view(batch_size, 1, in_size, in_size),retain_graph=True)
        grad_clone[i,:,:,:] = adder_ref.adder.grad.clone()
        # adder_ref.adder.grad.zero_()
        # print(grad_clone[i,:,:,:])
    
    # adder_ref = adder2d(1,
    #                 1,
    #                 kernel_size,
    #                 stride,
    #                 padding,
    #                 bias = True).cuda()
    # adder_ref.adder.data.copy_(weight[0,:,:,:].view(1, 1, kernel_size, kernel_size))
    # adder_ref.b.data.copy_(bias[0])
    # # print(adder_ref.adder.data)
    # # print(input[:,0,:,:].size())
    # out = adder_ref(input[:,0,:,:].view(batch_size, 1, in_size, in_size))
    # out.backward(grad_output[:,0,:,:].view(batch_size, 1, in_size, in_size),retain_graph=True)
    # grad_clone[0,:,:,:] = adder_ref.adder.grad.clone()
    # adder_ref.adder.grad.zero_()
    # print(grad_clone[0,:,:,:])


    adder = Adder2D(in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding,
                    groups,
                    bias = False,
                    eta = 0.2).cuda()
    adder.adder.data.copy_(weight)
    # adder.b.data.copy_(bias)

    output = adder(input)
    output.backward(grad_output, retain_graph=True)
    grad_weight = adder.adder.grad.clone()
    adder.adder.grad.zero_()

    eps = 1e-12
    print("Grad weight: max error: {:.3e}".format(((grad_clone - grad_weight) / (grad_clone.abs() + eps)).abs().max()))
    # print("Grad weight: max error: {:.3e}".format(((grad_clone - grad_weight)).abs().max()))
    # print(grad_weight)
    # print(grad_clone)


def check_grad_weight_2():
    batch_size = 1
    in_channels = 64
    out_channels = 64
    in_size = 128
    kernel_size = 3
    padding = 1
    stride = 1
    groups = in_channels
    out_size = (in_size + 2 * padding - kernel_size) // stride + 1
    print(out_size)

    input = tt.randn(batch_size, in_channels, in_size, in_size).cuda()
    # print(input)
    input.requires_grad = True
    weight = tt.randn(out_channels, 1, kernel_size, kernel_size).cuda()
    weight.requires_grad = True
    bias = tt.randn(out_channels).cuda()
    grad_output = tt.randn(batch_size, out_channels, out_size, out_size).cuda()
    grad_clone = tt.randn(out_channels, 1, kernel_size, kernel_size).cuda()
    out_ref = tt.randn(batch_size, out_channels, out_size, out_size).cuda()


    adder_2 = Adder2D_2(in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding,
                    groups,
                    bias = True,
                    eta = 0.2).cuda()
    adder_2.adder.data.copy_(weight)
    adder_2.b.data.copy_(bias)

    output = adder_2(input)
    output.backward(grad_output, retain_graph=True)
    grad_clone = adder_2.adder.grad.clone()
    adder_2.adder.grad.zero_()


    adder = Adder2D(in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding,
                    groups,
                    bias = True,
                    eta = 0.2).cuda()
    adder.adder.data.copy_(weight)
    adder.b.data.copy_(bias)

    output = adder(input)
    output.backward(grad_output, retain_graph=True)
    grad_weight = adder.adder.grad.clone()
    adder.adder.grad.zero_()

    eps = 1e-12
    print("Grad weight: max error: {:.3e}".format(((grad_clone - grad_weight) / (grad_clone.abs() + eps)).abs().max()))
    # print("Grad weight: max error: {:.3e}".format(((grad_clone - grad_weight)).abs().max()))
    # print(grad_weight)
    # print(grad_clone)

def check_naive_clone():
    batch_size = 1
    in_channels = 1
    out_channels = 1
    in_size = 3
    kernel_size = 1
    padding = 0
    stride = 1
    out_size = (in_size + 2 * padding - kernel_size) // stride + 1
    print(out_size)

    input = tt.randn(batch_size, in_channels, in_size, in_size).cuda()
    weight = tt.randn(out_channels, in_channels, kernel_size, kernel_size).cuda()
    # bias = tt.randn(out_channels).cuda()
    output = tt.randn(batch_size, out_channels, out_size, out_size).cuda()

    result = adder_cuda.forward(input,
                               weight,
                               # bias,
                               output,
                               kernel_size, kernel_size,
                               stride, stride,
                               padding, padding)
    print(result)
    input.clone()
    weight.clone()
    # bias.clone()
    output.clone()

    # F.conv2d(input, weight, bias, padding=padding)

    # input.clone()


if __name__ == '__main__':
    check_forward()
    check_grad_in()
    check_grad_weight()
    # check_naive_clone()
