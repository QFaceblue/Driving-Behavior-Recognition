from torch import nn

__all__ = ['MobileNetV2']

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
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, norm_layer=None):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            norm_layer(out_planes),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, norm_layer=None):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class Sandglass(nn.Module):
    def __init__(self, inp, oup, stride, reduce_ratio, norm_layer=None):
        super(Sandglass, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp / reduce_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []

        layers.extend([
            # dw
            ConvBNReLU(inp, inp, stride=1, groups=inp, norm_layer=norm_layer),
            # pw-linear
            nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
            norm_layer(hidden_dim),
            # pw-relu6
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
            nn.ReLU6(inplace=True),
            # dw-liner
            nn.Conv2d(oup, oup, 3, stride, 1, groups=oup, bias=False),
            norm_layer(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class My_Sandglass(nn.Module):
    def __init__(self, inp, oup, stride, reduce_ratio, norm_layer=None):
        super(My_Sandglass, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.act = nn.ReLU6(inplace=True)
        hidden_dim = int(round(inp / reduce_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        self.dw1 = nn.Conv2d(inp, inp, 3, 1, 1, groups=inp, bias=False)
        self.bn1 = norm_layer(inp)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(inp, hidden_dim)
        self.pw1 = nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False)
        self.bn2 = norm_layer(hidden_dim)
        self.pw2 = nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False)
        self.bn3 = norm_layer(oup)
        self.dw2 = nn.Conv2d(oup, oup, 3, stride, 1, groups=oup, bias=False)
        self.bn4 = norm_layer(oup)

    def forward(self, x):

        y = self.dw1(x)
        b, c, _, _ = y.size()
        z = self.avg_pool(y).view(b, c)
        z = self.fc(z).view(b, -1, 1, 1)
        z = torch.clamp(z, 0, 1)
        y = self.bn1(y)
        y = self.act(y)

        y = self.pw1(y)
        y = self.bn2(y)
        y = y * z

        y = self.pw2(y)
        y = self.bn3(y)
        y = self.act(y)

        y = self. dw2(y)
        y = self.bn4(y)

        if self.use_res_connect:
            return x + y
        else:
            return y

def hard_sigmoid(x, inplace: bool = False):
    if inplace:
        return x.add_(3.).clamp_(0., 6.).div_(6.)
    else:
        return F.relu6(x + 3.) / 6.

class My_Sandglass_2(nn.Module):
    def __init__(self, inp, oup, stride, reduce_ratio, norm_layer=None):
        super(My_Sandglass_2, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.act = nn.ReLU6(inplace=True)
        hidden_dim = int(round(inp / reduce_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        self.dw1 = nn.Conv2d(inp, inp, 3, 1, 1, groups=inp, bias=False)
        self.bn1 = norm_layer(inp)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(inp, hidden_dim)
        self.pw1 = nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False)
        self.bn2 = norm_layer(hidden_dim)
        self.pw2 = nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False)
        self.bn3 = norm_layer(oup)
        self.dw2 = nn.Conv2d(oup, oup, 3, stride, 1, groups=oup, bias=False)
        self.bn4 = norm_layer(oup)

    def forward(self, x):

        y = self.dw1(x)
        b, c, _, _ = y.size()
        z = self.avg_pool(y).view(b, c)
        z = self.fc(z).view(b, -1, 1, 1)
        # z = torch.clamp(z, 0, 1)
        z = hard_sigmoid(z, inplace=True)
        y = self.bn1(y)
        y = self.act(y)

        y = self.pw1(y)
        y = self.bn2(y)
        y = y * z

        y = self.pw2(y)
        y = self.bn3(y)
        y = self.act(y)

        y = self. dw2(y)
        y = self.bn4(y)

        if self.use_res_connect:
            return x + y
        else:
            return y

class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel))

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        y = torch.clamp(y, 0, 1)
        return x * y

class MobileNetV2(nn.Module):
    def __init__(self,
                 num_classes=1000,
                 width_mult=1.0,
                 inverted_residual_setting=None,
                 round_nearest=8,
                 block=None,
                 norm_layer=None):
        """
        MobileNet V2 main class

        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenetv2
            norm_layer: Module specifying the normalization layer to use

        """
        super(MobileNetV2, self).__init__()

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

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
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(3, input_channel, stride=2, norm_layer=norm_layer)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, t, norm_layer=norm_layer))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            # nn.Linear(self.last_channel, num_classes),
            nn.Linear(output_channel, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x):
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        x = self.features(x)
        # Cannot use "squeeze" as batch-size can be 1 => must use reshape with x.shape[0]
        x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        x = self.classifier(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)

class MobileNetV2_sandglass(nn.Module):
    def __init__(self,
                 num_classes=1000,
                 width_mult=1.0,
                 inverted_residual_setting=None,
                 round_nearest=8,
                 block=None,
                 norm_layer=None):
        """
        MobileNet V2 main class

        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenetv2
            norm_layer: Module specifying the normalization layer to use

        """
        super(MobileNetV2_sandglass, self).__init__()

        if block is None:
            block = Sandglass

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

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
                # [6, 320, 1, 1],

            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(3, input_channel, stride=2, norm_layer=norm_layer)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * t * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, t, norm_layer=norm_layer))
                input_channel = output_channel
        features.extend(
            [ConvBNReLU(960, 960, stride=1, groups=960, norm_layer=norm_layer),
            # pw-linear
            nn.Conv2d(960, 320, 1, 1, 0, bias=False),
            norm_layer(320),]
        )
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x):
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        x = self.features(x)
        # Cannot use "squeeze" as batch-size can be 1 => must use reshape with x.shape[0]
        x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        x = self.classifier(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)

class MobileNeXt(nn.Module):
    def __init__(self,
                 num_classes=1000,
                 width_mult=1.0,
                 sandglass_setting=None,
                 round_nearest=8,
                 block=None,
                 norm_layer=None):
        """
        MobileNet V2 main class

        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            sandglass_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenetv2
            norm_layer: Module specifying the normalization layer to use

        """
        super(MobileNeXt, self).__init__()

        if block is None:
            block = Sandglass

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        input_channel = 32
        last_channel = 1280

        if sandglass_setting is None:
            sandglass_setting = [
                # t, c, n, s
                [2, 96, 1, 2],
                [6, 144, 1, 1],
                [6, 192, 3, 2],
                [6, 288, 3, 2],
                [6, 384, 4, 1],
                [6, 576, 4, 2],
                [6, 960, 3, 1],# [6, 960, 2, 1],
                [6, 1280, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(sandglass_setting) == 0 or len(sandglass_setting[0]) != 4:
            raise ValueError("sandglass_setting should be non-empty "
                             "or a 4-element list, got {}".format(sandglass_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(3, input_channel, stride=2, norm_layer=norm_layer)]
        # building sandglass blocks
        for t, c, n, s in sandglass_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, t, norm_layer=norm_layer))
                input_channel = output_channel
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x):
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        x = self.features(x)
        # Cannot use "squeeze" as batch-size can be 1 => must use reshape with x.shape[0]
        x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        x = self.classifier(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)

def mobilenetv2_sandglass(**kwargs):

    sandgrass_setting = [
        # t, c, n, s
        [1, 16, 1, 1],
        [6, 144, 2, 2],
        [6, 192, 3, 2],
        [6, 384, 4, 2],
        [6, 576, 3, 1],
        [6, 960, 3, 2],
        [6, 1920, 1, 1],

        # [1, 16, 1, 1],
        # [6, 24, 2, 2],
        # [6, 32, 3, 2],
        # [6, 64, 4, 2],
        # [6, 96, 3, 1],
        # [6, 160, 3, 2],
        # [6, 320, 1, 1],

    ]
    block = Sandglass
    return MobileNetV2(inverted_residual_setting=sandgrass_setting, block=block, **kwargs)

def my_mobilenext(**kwargs):

    block = My_Sandglass
    return MobileNeXt(block=block, **kwargs)

def my_mobilenext_2(**kwargs):

    block = My_Sandglass_2
    return MobileNeXt(block=block, **kwargs)

if __name__=='__main__':
    import torch
    from torchvision import models

    model = MobileNeXt()
    print('Total params: %f M' % (sum(p.numel() for p in model.parameters()) / 1024. / 1024.0))
    print(len(list(model.modules())))
    # model = my_mobilenext()
    # print('Total params: %f M' % (sum(p.numel() for p in model.parameters()) / 1024. / 1024.0))
    # print(len(list(model.modules())))
    # model = MobileNetV2()
    # print('Total params: %f M' % (sum(p.numel() for p in model.parameters()) / 1024. / 1024.0))
    # print(len(list(model.modules())))
    # model = models.mobilenet_v2(pretrained=False, width_mult=1.0)
    # print('Total params: %f M' % (sum(p.numel() for p in model.parameters()) / 1024. / 1024.0))
    # print(len(list(model.modules())))
    # model =mobilenetv2_sandglass()
    # print('Total params: %f M' % (sum(p.numel() for p in model.parameters()) / 1024. / 1024.0))
    # print(len(list(model.modules())))
    # model = MobileNetV2_sandglass()
    # print('Total params: %f M' % (sum(p.numel() for p in model.parameters()) / 1024. / 1024.0))
    # print(len(list(model.modules())))
    # model = InvertedResidual(32, 32, 1, 6)
    # print('InvertedResidual params: %.f' % (sum(p.numel() for p in model.parameters())))
    # print(len(list(model.modules())))
    # print(model)
    # model = Sandglass(192, 192, 1, 6)
    # print('Sandglass params: %.f' % (sum(p.numel() for p in model.parameters())))
    # print(len(list(model.modules())))
    # # print(model)
    # model = My_Sandglass(192, 192, 1, 6)
    # print('Sandglass params: %.f' % (sum(p.numel() for p in model.parameters())))
    # print(len(list(model.modules())))
    # print(model)
    # model.eval()
    # # print(model)
    input = torch.randn(1, 3, 224, 224)
    # y = model(input)
    # # print(y.shape)
    # print('Total params: %f M' % (sum(p.numel() for p in model.parameters())/ 1024. / 1024.0))
    from thop import profile
    flops, params = profile(model, inputs=[input])
    print(flops)
    print(params)
    print('Total params: %f M' % (sum(p.numel() for p in model.parameters())))