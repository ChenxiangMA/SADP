import torch
from torch import nn
# from spikingjelly.activation_based.layer import Dropout
cfgs = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512],
    'VGG11': [128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
}

class VGG(nn.Module):
    def __init__(self, cfg, num_classes=10, init_weights=True, spiking_neuron=None):
        super(VGG, self).__init__()
        self.norm_layer = nn.BatchNorm2d
        self.cfg = cfg
        self.features = self.make_layers(cfg=cfgs[cfg], spiking_neuron=spiking_neuron)
        if self.cfg == 'A':
            self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
            self.fc = nn.Linear(512*7*7, num_classes)
        else:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

            self.fc = nn.Linear(512, num_classes)
        if init_weights:
            self._initialize_weights()

    def forward(self, x, time_step=0):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, self.norm_layer):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def make_layers(self, cfg, spiking_neuron):
        layers = []
        if self.cfg == 'VGG11' or self.cfg == 'A':
            in_channels = 2
        else:
            in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
            else:
                layers += [BasicLayer(in_channels, x, spiking_neuron)]
                in_channels = x
        return nn.Sequential(*layers)

class BasicLayer(nn.Module):

    def __init__(self, in_channels, out_channels, spiking_neuron):
        super(BasicLayer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.spk = spiking_neuron()
        self.dropout = Dropout(0.3)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.spk(x)
        return x


def _vgg(arch, cfg, spiking_neuron=None, batch_norm=True, norm_layer=None, **kwargs):
    model = VGG(cfg=cfg, spiking_neuron=spiking_neuron, **kwargs)
    return model


def vgg11(spiking_neuron=None, batch_norm=True, norm_layer=None, class_num=10, **kwargs):
    return _vgg("vgg11", 'VGG11', spiking_neuron=spiking_neuron, batch_norm=batch_norm, norm_layer=norm_layer, num_classes=class_num)






