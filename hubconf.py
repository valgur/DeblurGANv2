from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from future import standard_library

standard_library.install_aliases()

dependencies = ['torch', 'torchvision', 'pretrainedmodels', 'numpy', 'yaml', 'future']

import torch.hub

from util.predictor import Predictor as _Predictor, load_model as _load_model

_pretrained_urls = {
    'fpn_inception': 'https://github.com/valgur/DeblurGANv2/releases/download/v1.0/DeblurGANv2-fpn_inception-0ab85a04.pt',
    'fpn_mobilenet': 'https://github.com/valgur/DeblurGANv2/releases/download/v1.0/DeblurGANv2-fpn_mobilenet-6d1b6d98.pt',
}


def DeblurGANv2(pretrained_name=None, progress=True, map_location=None, **model_config):
    if not pretrained_name:
        return _load_model(weights=None, **model_config)

    url = _pretrained_urls[pretrained_name]
    weights = torch.hub.load_state_dict_from_url(url, map_location=map_location, progress=progress, check_hash=True)
    model_config['g_name'] = pretrained_name
    model_config['pretrained'] = False  # do not load imagenet weights
    model = _load_model(weights, **model_config)
    return model


def predictor(pretrained_name, device='cuda', **model_config):
    model = DeblurGANv2(pretrained_name, map_location=device, **model_config)
    return _Predictor(model, device=device)
