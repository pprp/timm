import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedPool_l(nn.Module):
    def __init__(self, kernel_size, stride, padding, dilation=1):
        super(GatedPool_l, self).__init__()
        self.mask = nn.Parameter(torch.rand(1, 1, kernel_size, kernel_size))
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.return_indices = False
        self.ceil_mode = False

    def forward(self, x):
        size = list(x.size())[1]
        out_size = list(x.size())[2] // 2
        bs = list(x.size())[0]
        xc = []

        for i in range(size):
            a = x[:, i, :, :]
            a = torch.unsqueeze(a, 1)
            a = F.conv2d(a, self.mask, stride=self.stride)
            xc.append(a)

        output = xc[0]

        for i in xc[1:]:
            output = torch.cat((output, i), 1)

        alpha = torch.sigmoid(output)

        x = alpha * F.max_pool2d(x, self.kernel_size, self.stride, self.padding, self.dilation) + (1-alpha) * \
            F.avg_pool2d(x, self.kernel_size, self.stride, self.padding)
        return x


class GatedPool_c(nn.Module):
    def __init__(self, in_channels, kernel_size, stride, padding=0, dilation=1):
        super(GatedPool_c, self).__init__()
        out_channel = in_channels
        self.mask = nn.Parameter(torch.randn(in_channels, out_channel, kernel_size, kernel_size))
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x):
        size = list(x.size())[1]
        out_size = list(x.size())[2] // 2
        bs = list(x.size())[0]
        mask_c = F.conv2d(x, self.mask, stride=self.stride)
        alpha = torch.sigmoid(mask_c)

        x = alpha * F.max_pool2d(x, self.kernel_size, self.stride, self.padding, self.dilation) + (1 - alpha) * \
            F.avg_pool2d(x, self.kernel_size, self.stride, self.padding)
        return x