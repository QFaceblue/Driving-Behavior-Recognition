import torch.nn as nn
import torch
import math

__all__ = ['mnext',]


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


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

def group_conv_1x1_bn(inp, oup, expand_ratio):
    hidden_dim = oup // expand_ratio
    return nn.Sequential(
        nn.Conv2d(inp, hidden_dim, 1, 1, 0, groups=hidden_dim, bias=False),
        nn.BatchNorm2d(hidden_dim),
        nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

class SGBlock(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, keep_3x3=False):
        super(SGBlock, self).__init__()
        assert stride in [1, 2]

        hidden_dim = inp // expand_ratio
        if hidden_dim < oup / 6.:
            hidden_dim = math.ceil(oup / 6.)
            hidden_dim = _make_divisible(hidden_dim, 16)# + 16

        #self.relu = nn.ReLU6(inplace=True)
        self.identity = False
        self.identity_div = 1
        self.expand_ratio = expand_ratio
        if expand_ratio == 2:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(inp, inp, 3, 1, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(oup, oup, 3, stride, 1, groups=oup, bias=False),
                nn.BatchNorm2d(oup),
            )
        elif inp != oup and stride == 1 and keep_3x3 == False:
            self.conv = nn.Sequential(
                # pw-linear
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True),
            )
        elif inp != oup and stride == 2 and keep_3x3==False:
            self.conv = nn.Sequential(
                # pw-linear
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(oup, oup, 3, stride, 1, groups=oup, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            if keep_3x3 == False:
                self.identity = True
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(inp, inp, 3, 1, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU6(inplace=True),
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                #nn.ReLU6(inplace=True),
                # pw
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(oup, oup, 3, 1, 1, groups=oup, bias=False),
                nn.BatchNorm2d(oup),
            )

    # def forward(self, x):
    #     out = self.conv(x)
    #
    #     if self.identity:
    #         shape = x.shape
    #         id_tensor = x[:,:shape[1]//self.identity_div,:,:]
    #         # id_tensor = torch.cat([x[:,:shape[1]//self.identity_div,:,:],torch.zeros(shape)[:,shape[1]//self.identity_div:,:,:].cuda()],dim=1)
    #         # import pdb; pdb.set_trace()
    #         out[:,:shape[1]//self.identity_div,:,:] = out[:,:shape[1]//self.identity_div,:,:] + id_tensor
    #         return out #+ x
    #     else:
    #         return out

    def forward(self, x):
        out = self.conv(x)

        if self.identity:
            return out + x
        else:
            return out

class MXNet(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.):
        super(MXNet, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = [
            # t, c, n, s
            [2,  96, 1, 2],
            [6, 144, 1, 1],
            [6, 192, 3, 2],
            [6, 288, 3, 2],
            [6, 384, 4, 1],
            [6, 576, 4, 2],
            [6, 960, 3, 1],
            [6, 1280, 1, 1],
        ]
        #self.cfgs = [
        #    # t, c, n, s
        #    [1,  16, 1, 1],
        #    [4,  24, 2, 2],
        #    [4,  32, 3, 2],
        #    [4,  64, 3, 2],
        #    [4,  96, 4, 1],
        #    [4, 160, 3, 2],
        #    [4, 320, 1, 1],
        #]

        # building first layer
        input_channel = _make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]
        # building inverted residual blocks
        block = SGBlock
        for t, c, n, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)
            if c == 1280 and width_mult < 1:
                output_channel = 1280
            layers.append(block(input_channel, output_channel, s, t, n==1 and s==1))
            input_channel = output_channel
            for i in range(n-1):
                layers.append(block(input_channel, output_channel, 1, t))
                input_channel = output_channel
        self.features = nn.Sequential(*layers)
        # building last several layers
        input_channel = output_channel
        output_channel = _make_divisible(input_channel, 4) # if width_mult == 0.1 else 8) if width_mult > 1.0 else input_channel
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(output_channel, num_classes)
                )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        #x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def mnext(**kwargs):
    """
    Constructs a mobilenext model
    """
    return MXNet(**kwargs)

if __name__=='__main__':
    import torch
    from torchvision import models
    model = mnext()
    print('Total params: %f M' % (sum(p.numel() for p in model.parameters()) / 1024. / 1024.0))
    print(len(list(model.modules())))

    input = torch.randn(1, 3, 224, 224)
    # y = model(input)
    # # print(y.shape)
    # print('Total params: %f M' % (sum(p.numel() for p in model.parameters())/ 1024. / 1024.0))
    from thop import profile
    flops, params = profile(model, inputs=[input])
    print(flops)
    print(params)
    print('Total params: %f M' % (sum(p.numel() for p in model.parameters())))