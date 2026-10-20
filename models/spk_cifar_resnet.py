import torch.nn as nn



def conv3x3(in_planes, out_planes, stride=1):
    " 3x3 convolution with padding "
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dropout_rate=0, spiking_neuron=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.spk1 = spiking_neuron(bn=self.bn1)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.spk2 = spiking_neuron(bn=self.bn2)
        self.downsample = downsample
        self.stride = stride
        # self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.spk1(out)
        # out = self.dropout(out)
        out = self.conv2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out = out + residual
        out = self.spk2(out)
        return out


class SEW_BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dropout_rate=0, spiking_neuron=None):
        super(SEW_BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.spk1 = spiking_neuron(bn=self.bn1)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.spk2 = spiking_neuron(bn=self.bn2)
        self.downsample = downsample
        if self.downsample is not None:
            self.downsample_spk = spiking_neuron()
        self.stride = stride
        # self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.spk1(out)

        # out = self.dropout(out)

        out = self.conv2(out)
        out = self.spk2(out)
        if self.downsample is not None:
            residual = self.downsample_spk(self.downsample(x))

        out = residual + out

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dropout_rate=0, spiking_neuron=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.spk1 = spiking_neuron(bn=self.bn1)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.spk2 = spiking_neuron(bn=self.bn2)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.spk3 = spiking_neuron(bn=self.bn3)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.spk1(out)

        out = self.conv2(out)
        out = self.spk2(out)

        out = self.conv3(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.spk3(out)

        return out

class SEW_Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dropout_rate=0, spiking_neuron=None):
        super(SEW_Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.spk1 = spiking_neuron(bn=self.bn1)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.spk2 = spiking_neuron(bn=self.bn2)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.spk3 = spiking_neuron(bn=self.bn3)
        self.downsample = downsample
        if self.downsample is not None:
            self.downsample_spk = spiking_neuron()
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.spk1(out)

        out = self.conv2(out)
        out = self.spk2(out)

        out = self.conv3(out)
        out = self.spk3(out)
        if self.downsample is not None:
            residual = self.downsample_spk(self.downsample(x))

        out = out + residual
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, class_num=10,
                 wide_list=(16, 16, 32, 64), dropout_rate=0, spiking_neuron=None, use_pool=False, args=None
                 ):
        super(ResNet, self).__init__()
        self.inplanes = wide_list[0]
        self.dropout_rate = dropout_rate
        self.feature_num = wide_list[-1]
        self.class_num = class_num
        self.layers = layers
        assert len(self.layers) in [3, 4]
        if args.dataset in ['cifar10dvs', 'hardvs', 'gaitday', 'dvsgesture', 'gaitnight']:
            data_num_channels = 2
        else:
            data_num_channels = 3
        self.conv1 = nn.Conv2d(data_num_channels, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.spk1 = spiking_neuron(bn=self.bn1)
        self.use_pool = use_pool
        if self.use_pool:
            self.maxpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, wide_list[1], layers[0], spiking_neuron=spiking_neuron)
        self.layer2 = self._make_layer(block, wide_list[2], layers[1], stride=2, spiking_neuron=spiking_neuron)
        self.layer3 = self._make_layer(block, wide_list[3], layers[2], stride=2, spiking_neuron=spiking_neuron)
        if len(self.layers) == 4:
            self.layer4 = self._make_layer(block, wide_list[4], layers[3], stride=2, spiking_neuron=spiking_neuron)
        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                        nn.Flatten(),
                                        nn.Linear(self.feature_num * block.expansion, self.class_num))

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, spiking_neuron=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, dropout_rate=self.dropout_rate,
                            spiking_neuron=spiking_neuron))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dropout_rate=self.dropout_rate, spiking_neuron=spiking_neuron))

        return nn.Sequential(*layers)

    def forward(self, img):

        x = self.conv1(img)
        x = self.spk1(x)
        if self.use_pool:
            x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if len(self.layers) == 4:
            x = self.layer4(x)
        x = self.classifier(x)
        return x





def spk_resnet18(**kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], wide_list=(64, 64, 128, 256, 512), **kwargs)
    return model

def spk_resnet34(**kwargs):
    model = ResNet(SEW_BasicBlock, [3, 4, 6, 3], wide_list=(64, 64, 128, 256, 512), **kwargs)
    return model

def spk_resnet19(**kwargs):
    model = ResNet(BasicBlock, [3, 3, 2], wide_list=(128, 128, 256, 512), **kwargs)
    # model = ResNet(BasicBlock, [5, 5, 5], wide_list=(128, 128, 256, 512), **kwargs)
    return model

def spk_resnet110(**kwargs):
    model = ResNet(SEW_BasicBlock, [18, 18, 18], wide_list=(64, 64, 128, 256), **kwargs)
    return model
def spk_resnet101(**kwargs):
    model = ResNet(SEW_Bottleneck, [3, 4, 23, 3], wide_list=(64, 64, 128, 256, 512), **kwargs)
    return model


def spk_resnet32(**kwargs):
    model = ResNet(BasicBlock, [5, 5, 5], **kwargs)
    return model





