import os
import pickle
import random
import string
import json
import logging
from pathlib import Path

from omegaconf import OmegaConf
import numpy as np
import PIL.Image as Image
from torch.utils.data import Dataset

WHITE = 255

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]


class FTGANDataset(Dataset):
    def __init__(self, args, mode='train'):
        super(FTGANDataset, self).__init__()
        self.args = args
        self.mode = mode

        if self.mode == 'train':
            self.font_dir = Path(args.font_dir) / 'train'
        else:
            if args.test_unknown_content:
                self.font_dir = Path(args.font_dir) / 'test_unknown_content'
            else:
                self.font_dir = Path(args.font_dir) / 'test_unknown_style'

        # Chinese(content) to English(style)

        self.content_root = self.font_dir / 'chinese'
        self.style_root = self.font_dir / 'english'
        self.source_root = self.font_dir / 'source'
        self.paths = []

        for img_ext in IMG_EXTENSIONS:
            self.paths += [x for x in self.content_root.glob(f"**/*{img_ext}")]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        """FTGANDataset __getitem__

        Args:
            idx (int): torch dataset index

        Returns:
            dict: return dict with following keys

            gt_images: target_image,
            content_images: same_chars_image,
            style_images: same_fonts_image,
            style_idx: font_idx,
            char_idx: char_idx,
            content_image_idxs: same_chars,
            style_image_idxs: same_fonts,
            image_paths: ''
        """

        gt_path = self.paths[index]
        font_dirname = gt_path.parent.name
        char_filename = gt_path.name

        style_font_dir = self.style_root / font_dirname

        style_font_path_list = [x for x in style_font_dir.iterdir() if x.is_file()]

        style_paths = random.sample(style_font_path_list, self.args.reference_imgs.style)
        content_path = self.source_root / char_filename

        content_image = Image.open(content_path).convert('RGB').resize((self.args.imsize, self.args.imsize))
        gt_image = Image.open(gt_path).convert('RGB').resize((self.args.imsize, self.args.imsize))
        style_images = [Image.open(x).convert('RGB').resize((self.args.imsize, self.args.imsize))
                        for x in style_paths]

        content_fonts_image = np.mean(np.array(content_image, dtype=np.float32), axis=-1)[np.newaxis, ...] / WHITE
        style_chars_image = np.array([np.mean(np.array(x), axis=-1) / WHITE
                                      for x in style_images], dtype=np.float32)
        target_image = np.mean(np.array(gt_image, dtype=np.float32), axis=-1)[np.newaxis, ...] / WHITE

        dict_return = {
            # data for training
            'gt_images': target_image,
            'content_images': content_fonts_image,
            'style_images': style_chars_image,
            # data for logging
            'style_idx': font_dirname,
            'char_idx': char_filename,
            'content_image_idxs': '',
            'style_image_idxs': '',
            'image_paths': '',
        }
        return dict_return


if __name__ == '__main__':
    hp = OmegaConf.load('config/datasets/ftgan.yaml').datasets.train
    _dataset = FTGANDataset(hp, font_dir=FONT_DIR)
    TEST_ITER_NUM = 4
    for i in range(TEST_ITER_NUM):
        data = _dataset[i]
        print(data.keys())
        print(data['gt_images'].shape,
              data['content_images'].shape,
              data['style_images'].shape,
              data['style_idx'],
              data['char_idx'])
