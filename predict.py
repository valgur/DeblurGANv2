from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from future import standard_library

standard_library.install_aliases()
from builtins import *
import os
from glob import glob

import cv2
import numpy as np
from fire import Fire
from tqdm import tqdm

from hubconf import DeblurGANv2
from util.predictor import Predictor, load_model


def main(img_pattern,
         mask_pattern = None,
         pretrained_model=None,
         weights_path='best_fpn.h5',
         out_dir='submit/',
         side_by_side = False,
         device='cuda'):
    def sorted_glob(pattern):
        return sorted(glob(pattern))

    imgs = sorted_glob(img_pattern)
    masks = sorted_glob(mask_pattern) if mask_pattern is not None else [None for _ in imgs]
    pairs = zip(imgs, masks)
    names = sorted([os.path.basename(x) for x in glob(img_pattern)])
    if pretrained_model:
        model = DeblurGANv2(pretrained_model, map_location=device)
    else:
        model = load_model(weights_path, map_location=device)
    predictor = Predictor(model, device=device)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for name, pair in tqdm(zip(names, pairs), total=len(names)):
        f_img, f_mask = pair
        img, mask = map(cv2.imread, (f_img, f_mask))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        pred = predictor(img, mask)
        if side_by_side:
            pred = np.hstack((img, pred))
        pred = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(out_dir, name), pred)


if __name__ == '__main__':
    Fire(main)
