import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Parameter
from torch.autograd import Variable

import math

# SGE module
class SpatialGroupEnhance(nn.Module):

    def __init__(self, groups = 64):
        super(SpatialGroupEnhance, self).__init__()
        self.groups   = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sig      = nn.Sigmoid()

        self.weight   = Parameter(torch.zeros(1, groups, 1, 1))
        self.bias     = Parameter(torch.ones(1, groups, 1, 1))

    def forward(self, x): 
        """ x: (b, c, h, w) """
        b, c, h, w = x.size()
        x = x.view(b * self.groups, -1, h, w) 
        xn = x * self.avg_pool(x)
        xn = xn.sum(dim=1, keepdim=True)
        t = xn.view(b * self.groups, -1)
        t = t - t.mean(dim=1, keepdim=True)
        std = t.std(dim=1, keepdim=True) + 1e-5
        t = t / std
        t = t.view(b, self.groups, h, w)
        t = t * self.weight + self.bias
        t = t.view(b * self.groups, 1, h, w)
        x = x * self.sig(t)
        x = x.view(b, c, h, w)
        return x

class Bottleneck(nn.Module):
    def __init__(self, inp, oup, stride, expansion, sgegroup):
        """
        inp: input channels
        oup: output channels
        stride: stride
        expansion: expansion
        """
        super(Bottleneck, self).__init__()
        
        self.connect = (stride == 1 and inp == oup)

        sgegroup = (64 if sgegroup is None else sgegroup)
        hidden_dim = int(inp * expansion)
        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.PReLU(hidden_dim),

            # dw
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.PReLU(hidden_dim),

            # sge
            SpatialGroupEnhance(sgegroup) if sgegroup > 0 else nn.Sequential(),

            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        out = self.conv(x)

        if self.connect:
            return x + out

        return out

class ConvBlock(nn.Module):
    def __init__(self, inp, oup, k, s, p, dw=False, linear=False):
        """
        inp: input channels
        oup: output channels
        k: kernel size
        s: stride
        p: padding
        """
        super(ConvBlock, self).__init__()

        groups      = inp if dw else 1
        self.conv   = nn.Conv2d(inp, oup, k, s, p, groups=groups, bias=False)
        self.bn     = nn.BatchNorm2d(oup)
        
        self.linear = (True if linear else False)
        if not self.linear:
            self.prelu  = nn.PReLU(oup)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)

        return x if self.linear else self.prelu(x)

"""
t: expansion factor
c: output channels
n: repeated n times
s: stride of the first layer of each sequence
"""
Mobilefacenet_bottleneck_setting = [
    #t, c,   n, s
    [2, 64,  5, 2],
    [4, 128, 1, 2],
    [2, 128, 6, 1],
    [4, 128, 1, 2],
    [2, 128, 2, 1]
]

class MobileFacenet(nn.Module):
    def __init__(self, block=Bottleneck, bottleneck_setting=Mobilefacenet_bottleneck_setting, sgegroup=64):
        super(MobileFacenet, self).__init__()
        self.conv3x3     = ConvBlock(3, 64, 3, 2, 1)
        self.dw_conv3x3  = ConvBlock(64, 64, 3, 1, 1, dw=True)
        self.bottlenecks = self._make_layer(block, bottleneck_setting, sgegroup)
        self.conv1x1     = ConvBlock(128, 512, 1, 1, 0)
        self.linear7x7   = ConvBlock(512, 512, (7, 6), 1, 0, dw=True, linear=True)
        self.linear1x1   = ConvBlock(512, 128, 1, 1, 0, linear=True)

        self._initialize_weights()

    def _make_layer(self, block, setting, sgegroup):
        layers = []
        
        # a fixed value for MobileFacenet
        input_channel = 64
        for t, c, n, s in setting:
            for i in range(n):
                stride = (s if i == 0 else 1)
                layers.append(block(input_channel, c, stride, t, sgegroup))
                input_channel = c

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv3x3(x)
        x = self.dw_conv3x3(x)
        x = self.bottlenecks(x)
        x = self.conv1x1(x)
        x = self.linear7x7(x)
        x = self.linear1x1(x)
        x = x.view(x.size(0), -1)

        return x

if __name__ == "__main__":
    input = Variable(torch.FloatTensor(2, 3, 112, 96))
    net = MobileFacenet(sgegroup=0)
    print(net)
    x = net(input)
    print(x.shape)

    # net = MobileFacenet(sge)
    # print(sge)
    # # (b, c, h, w)
    # input = torch.FloatTensor(2, 64, 112, 96)
    # x = sge(input)
    # print(x.shape)

