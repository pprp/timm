import torch
from torch import nn

from .registry import register_model

__all__ = ['mobilenet_v2']


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class SpatialSeperablePooling(nn.Module):
    """Spatial Seperable Pooling operation."""

    def __init__(self, in_channels, out_channels=None, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        if out_channels is None:
            out_channels = in_channels

        mid_channels = max(8, in_channels // reduction)

        self.conv1 = nn.Conv2d(in_channels, mid_channels,
                               kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(
            mid_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(
            mid_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self,
                 in_planes,
                 out_planes,
                 kernel_size=3,
                 stride=1,
                 groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes,
                      out_planes,
                      kernel_size,
                      stride,
                      padding,
                      groups=groups,
                      bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True))


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class MetaPooling(nn.Module):
    def __init__(self, in_channels=128):
        super().__init__()
        self.candidates = nn.ModuleList()
        self.candidates.append(SpatialSeperablePooling(in_channels))
        self.candidates.append(nn.AvgPool2d(3, stride=1, padding=1))
        self.candidates.append(nn.AvgPool2d(5, stride=1, padding=2))
        self.candidates.append(nn.AvgPool2d(7, stride=1, padding=3))
        self.candidates.append(nn.Identity())
        self.bn = nn.BatchNorm2d(in_channels)
        self.act = h_swish()

    def forward(self, x, index=0):
        out = self.candidates[index](x)
        out = self.bn(out)
        out = self.act(out)
        return out


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio, idx=6):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(in_channels * expand_ratio))
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(in_channels, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim,
                       hidden_dim,
                       stride=stride,
                       groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
        ])
        self.conv = nn.Sequential(*layers)
        self.meta = nn.Identity()
        # SpatialSeperablePooling(out_channels) if idx >= 6 else nn.Identity()

    def forward(self, x):
        if self.use_res_connect:
            return x + self.meta(self.conv(x))
        else:
            return self.meta(self.conv(x))


class MobileNetV2(nn.Module):
    def __init__(self,
                 num_classes=1000,
                 width_mult=1.0,
                 inverted_residual_setting=None,
                 round_nearest=8,
                 **kwargs):
        """
        MobileNet V2 main class

        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
        """
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(
                inverted_residual_setting[0]) != 4:
            raise ValueError('inverted_residual_setting should be non-empty '
                             'or a 4-element list, got {}'.format(
                                 inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult,
                                        round_nearest)
        self.last_channel = _make_divisible(
            last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(3, input_channel, stride=2)]
        # building inverted residual blocks
        global_cnt = 0
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(
                    block(input_channel,
                          output_channel,
                          stride,
                          expand_ratio=t,
                          idx=global_cnt))
                input_channel = output_channel
                global_cnt += 1
        # building last several layers

        features.append(
            ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        # make it nn.Sequential
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.features = nn.Sequential(*features)
        self.hwish = h_swish()
        self.fc = nn.Conv2d(self.last_channel, 1280, 1, 1, 0)
        self.dropout = nn.Dropout(0.2)

        # building classifier
        self.classifier = nn.Linear(1280, num_classes)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.fc(x)
        x = self.hwish(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


@register_model
def mobilenet_v2(pretrained=False, **kwargs):
    """
    Constructs a MobileNetV2 architecture from
    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return MobileNetV2(**kwargs)


@register_model
def mobilenet_v2_075(pretrained=False, **kwargs):
    """
    Constructs a MobileNetV2 architecture from
    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return MobileNetV2(width_mult=0.75, **kwargs)


@register_model
def mobilenet_v2_050(pretrained=False, **kwargs):
    """
    Constructs a MobileNetV2 architecture from
    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return MobileNetV2(width_mult=0.5, **kwargs)


def demo():
    net = mobilenet_v2(num_classes=1000)
    y = net(torch.randn(2, 3, 224, 224))
    print(y.size())


if __name__ == '__main__':
    demo()
