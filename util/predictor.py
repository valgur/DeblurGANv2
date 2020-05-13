import os
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import yaml

from models.networks import get_generator


def load_default_config():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(script_dir, '../config/config.yaml')) as cfg:
        return yaml.safe_load(cfg)


def load_model(weights: Union[str, dict] = None, map_location=None, **model_config) -> nn.Module:
    full_model_config = load_default_config()['model']
    full_model_config.update(model_config)
    model = get_generator(full_model_config)
    if weights:
        if not isinstance(weights, dict):
            weights = torch.load(weights, map_location=map_location)
        model.load_state_dict(weights['model'])
    # Remove DataParallel for CPU inference support
    model = model.module
    # GAN inference should be in train mode to use actual stats in norm layers,
    # it's not a bug
    model.train(True)
    return model


class Predictor:
    def __init__(self, model: nn.Module, device='cuda'):
        self.device = torch.device(device)
        self.model = model.to(self.device)

    @staticmethod
    def _array_to_batch(x):
        x = np.transpose(x, (2, 0, 1))
        x = np.expand_dims(x, 0)
        return torch.from_numpy(x)

    def _preprocess(self, x: np.ndarray, mask: Optional[np.ndarray]) -> (torch.Tensor, torch.Tensor, int, int):
        x = (x.astype(np.float32) - 127.5) / 127.5
        if mask is None:
            mask = np.ones_like(x, dtype=np.float32)
        else:
            mask = np.round(mask.astype('float32') / 255)

        h, w, _ = x.shape
        block_size = 32
        min_height = (h // block_size + 1) * block_size
        min_width = (w // block_size + 1) * block_size

        pad_params = {'mode': 'constant',
                      'constant_values': 0,
                      'pad_width': ((0, min_height - h), (0, min_width - w), (0, 0))
                      }
        x = np.pad(x, **pad_params)
        mask = np.pad(mask, **pad_params)

        return self._array_to_batch(x), self._array_to_batch(mask), h, w

    @staticmethod
    def _postprocess(x: torch.Tensor) -> np.ndarray:
        x, = x
        x = x.detach().cpu().float().numpy()
        x = (np.transpose(x, (1, 2, 0)) + 1) / 2.0 * 255.0
        return x.astype('uint8')

    @torch.no_grad()
    def __call__(self, img: np.ndarray, mask: Optional[np.ndarray], ignore_mask=True) -> np.ndarray:
        img, mask, h, w = self._preprocess(img, mask)
        inputs = [img.to(self.device)]
        if not ignore_mask:
            inputs += [mask.to(self.device)]
        pred = self.model(*inputs)
        return self._postprocess(pred)[:h, :w, :]
