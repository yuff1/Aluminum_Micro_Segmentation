# Copyright (c) OpenMMLab. All rights reserved.
import os
from argparse import ArgumentParser
import numpy as np
import mmcv
import sys

from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
import torch
import cv2
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from torchvision import transforms
from PIL import Image


def main():
    parser = ArgumentParser()
    parser.add_argument('--my_config', default="./my_config/resunet_ca.py", help='Config file')
    parser.add_argument('--checkpoint', default="./work_dirs/resunet_ca/iter_20000.pth", help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='my',
        help='Color palette used for segmentation map')
    parser.add_argument(
        '--opacity',
        type=float,
        default=1,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    args = parser.parse_args()
    seg_model = init_segmentor(args.my_config, args.checkpoint, device=args.device)
    test_path = "./tt/jx/"
    save_path = "./tt/jx2/"
    file_list = os.listdir(test_path)

    for file in tqdm(file_list[0:200]):
        result = inference_segmentor(seg_model, os.path.join(test_path, file))
        palette = [[0, 0, 0], [255, 255, 255]]
        show_result_pyplot(
            seg_model,
            os.path.join(test_path, file),
            result,
            palette,
            opacity=args.opacity,
            out_file=os.path.join(save_path, file))


if __name__ == '__main__':
    main()
