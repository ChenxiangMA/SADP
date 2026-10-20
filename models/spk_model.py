from functools import partial

import torch.nn as nn
import torch

from models import spk_cifar_resnet, spk_cifar_vgg
from spiking_neuron import LIFNode
from surrogate_gradient import TriangleSurroGrad
from tools.utils import reset_states



class SpkModel(nn.Module):
    def __init__(self, args=None):
        super(SpkModel, self).__init__()
        self.time_window = args.time_window
        self.dataset = args.dataset

        spiking_neuron = partial(LIFNode,
                                    decay_factor=args.decay,
                                    v_threshold=args.threshold,
                                    surrogate_function=TriangleSurroGrad.apply,
                                    hard_reset=False,
                                    detach_reset=True,
                                    detach_mem=False,
                                    pos_bn=args.pos_bn)
        if self.dataset in ['hardvs', 'imagenet']:
            self.encoder = eval(args.arch + '.' + args.net)(spiking_neuron=spiking_neuron, class_num=args.num_classes,
                                                            use_pool=True, args=args)
        else:
            self.encoder = eval(args.arch + '.' + args.net)(spiking_neuron=spiking_neuron, class_num=args.num_classes,
                                                            use_pool=False, args=args)

    def forward(self, img):
        reset_states(model=self.encoder)
        for t in range(self.time_window):
            if self.dataset in ['cifar10dvs', 'hardvs']:
                logits = self.encoder(img[t])
            else:
                logits = self.encoder(img)
            if t == 0:
                output = logits.unsqueeze(1)
            else:
                output = torch.cat((output, logits.unsqueeze(1)), dim=1)

        return output






