import functools

import torch
from torch import nn
from .ssd import SSD_CONFIG, SingleShotDetector
from .basenet import load_from_file_or_model_zoo


def add_save_and_load(create_func):
    def extended_create_func(*args, **kwargs):
        net = create_func(*args, **kwargs)
        net.save = functools.partial(save, net=net, args=args, kwargs=kwargs)
        net.load = load
        return net
    return extended_create_func


REGISTERED_MODEL = {}


def register_model(arch_prefix, model):
    REGISTERED_MODEL[arch_prefix] = model


@add_save_and_load
def create(arch, classnames, basenet='vgg16', pretrained='imagenet', freeze_pretrained=0, frozen_bn=False):

    # compatibility for old models
    if classnames and classnames[0] == 'background':
        #print('Warning: removing "background" from classnames')
        classnames = classnames[1:]

    if arch in SSD_CONFIG.keys():
        net = SingleShotDetector(classnames, basenet=basenet, version=arch, pretrained=pretrained, frozen_bn=frozen_bn)
        net.set_pretrained_frozen(freeze_pretrained)
    elif arch.split(':')[0] == 'torchvision':
        from .torchvision_det import TorchVisionDet
        net = TorchVisionDet(arch.split(':')[1], classnames, pretrained=pretrained)
    elif arch.split(':')[0] in REGISTERED_MODEL:
        net = REGISTERED_MODEL[arch.split(':')[0]](arch.split(':')[1], classnames=classnames,
                                                   freeze_pretrained=freeze_pretrained,
                                                   frozen_bn=frozen_bn,
                                                   pretrained=pretrained)
    else:
        raise NotImplementedError(arch)

    return net


def save(filename, net, args, kwargs):
    if isinstance(net, nn.DataParallel):
        net = net.module

    data = dict(args=args,
                kwargs=kwargs,
                state_dict=net.state_dict())
    torch.save(data, filename)


def load(filename):
    print('load {}'.format(filename))

    if isinstance(filename, str) and (filename.startswith('torchvision:')
                                      or filename.startswith("detectron2:")
                                      or filename.startswith("mmdet:")):
        return create(filename, classnames=None, pretrained='coco')

    data = load_from_file_or_model_zoo(filename)

    if ('args' not in data) and ('kwargs' not in data):
        # loading old model
        net = SingleShotDetector(classnames=data['classes'], basenet=data['basenet'], version=data['version'],
                                 pretrained=False)
    else:
        data['kwargs']['pretrained'] = None
        net = create(*data['args'], **data['kwargs'])
    net.load_state_dict(data['state_dict'])
    return net
