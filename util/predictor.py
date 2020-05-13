import os
from typing import Optional

import numpy as np
import torch

from models.networks import get_generator


def load_default_config():
    import yaml
    script_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(script_dir, '../config/config.yaml')) as cfg:
        return yaml.safe_load(cfg)


class Predictor:
    def __init__(self, weights_path: str, model_config: Optional[dict] = None, device='cuda'):
        self.device = torch.device(device)
        model = get_generator(model_config or load_default_config()['model'])
        model.load_state_dict(torch.load(weights_path, map_location=self.device)['model'])
        # Remove DataParallel for CPU inference support
        model = model.module
        self.model = model.to(self.device)
        # GAN inference should be in train mode to use actual stats in norm layers,
        # it's not a bug
        self.model.train(True)

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
