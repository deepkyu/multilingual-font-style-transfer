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
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

REPEATE_NUM = 10000

WHITE = 255

MAX_TRIAL = 10

_upper_case = set(map(lambda s: f"{ord(s):04X}", string.ascii_uppercase))
_digits = set(map(lambda s: f"{ord(s):04X}", string.digits))
english_set = list(_upper_case.union(_digits))

NOTO_FONT_DIRNAME = "Noto"


class GoogleFontDataset(Dataset):
    def __init__(self, args, mode='train',
                 metadata_path="./lang_set.json"):
        super(GoogleFontDataset, self).__init__()
        self.args = args
        self.font_dir = Path(args.font_dir)
        self.mode = mode
        self.lang_list = sorted([x.stem for x in self.font_dir.iterdir() if x.is_dir()])
        self.min_tight_bound = 10000
        self.min_font_name = None

        if self.mode == 'train':
            self.lang_list = self.lang_list[:-2]
        else:
            self.lang_list = self.lang_list[-2:]
        with open(metadata_path, "r") as json_f:
            self.data = json.load(json_f)

        self.num_lang = None
        self.num_font = None
        self.num_char = None
        self.content_meta, self.style_meta, self.num_lang, self.num_font, self.num_char = self.get_meta()
        logging.info(f"min_tight_bound: {self.min_tight_bound}")  # 20

    @staticmethod
    def center_align(bg_img, item_img, fit=False):
        bg_img = bg_img.copy()
        item_img = item_img.copy()
        item_w, item_h = item_img.size
        W, H = bg_img.size
        if fit:
            item_ratio = item_w / item_h
            bg_ratio = W / H

            if bg_ratio > item_ratio:
                # height fitting
                resize_ratio = H / item_h
            else:
                # width fitting
                resize_ratio = W / item_w
            item_img = item_img.resize((int(item_w * resize_ratio), int(item_h * resize_ratio)))
            item_w, item_h = item_img.size

        bg_img.paste(item_img, ((W - item_w) // 2, (H - item_h) // 2))
        return bg_img

    def _get_content_image(self, png_path):
        im = Image.open(png_path)
        bg_img = Image.new('RGB', (self.args.imsize, self.args.imsize), color='white')
        blend_img = self.center_align(bg_img, im, fit=True)
        return blend_img

    def _get_style_image(self, png_path):
        im = Image.open(png_path)
        w, h = im.size

        # tight_bound_check & update
        tight_bound = self.get_tight_bound_size(np.array(im))
        if self.min_tight_bound > tight_bound:
            self.min_tight_bound = tight_bound
            self.min_font_name = png_path
            logging.debug(f"min_tight_bound: {self.min_tight_bound}, min_font_name: {self.min_font_name}")

        bg_img = Image.new('RGB', (max([w, h, self.args.imsize]), max([w, h, self.args.imsize])), color='white')
        blend_img = self.center_align(bg_img, im)
        return blend_img

    def get_meta(self):
        content_meta = dict()
        style_meta = dict()

        num_lang = 0
        num_font = 0
        num_char = 0
        for lang_dir in tqdm(self.lang_list, total=len(self.lang_list)):
            font_list = sorted([x for x in (self.font_dir / lang_dir).iterdir() if x.is_dir()])

            font_content_dict = dict()
            font_style_dict = dict()

            for font_dir in font_list:
                image_content_dict = dict()
                image_style_dict = dict()

                png_list = [x for x in font_dir.glob("*.png")]

                for png_path in png_list:

                    # image_content_dict[png_path.stem] = self._get_content_image(png_path)
                    # image_style_dict[png_path.stem] = self._get_style_image(png_path)
                    image_content_dict[png_path.stem] = png_path
                    image_style_dict[png_path.stem] = png_path
                    num_char += 1

                font_content_dict[font_dir.stem] = image_content_dict
                font_style_dict[font_dir.stem] = image_style_dict
                num_font += 1

            content_meta[lang_dir] = font_content_dict
            style_meta[lang_dir] = font_style_dict
            num_lang += 1

        return content_meta, style_meta, num_lang, num_font, num_char

    @staticmethod
    def get_tight_bound_size(img):
        contents_cell = np.where(img < WHITE)

        if len(contents_cell[0]) == 0:
            return 0

        size = {
            'xmin': np.min(contents_cell[1]),
            'ymin': np.min(contents_cell[0]),
            'xmax': np.max(contents_cell[1]) + 1,
            'ymax': np.max(contents_cell[0]) + 1,
        }
        return max(size['xmax'] - size['xmin'], size['ymax'] - size['ymin'])

    def get_patch_from_style_image(self, image, patch_per_image=1):
        w, h = image.size
        image_list = []
        relative_patch_size = int(self.args.imsize * 2)
        for _ in range(patch_per_image):
            offset = w - relative_patch_size
            if offset < relative_patch_size // 2:
                # if image is too small, just resize
                crop_candidate = np.array(image.resize((self.args.imsize, self.args.imsize)))
            else:
                # if image is sufficent to be cropped, randomly crop
                x = np.random.randint(0, offset)
                y = np.random.randint(0, offset)
                crop_candidate = image.crop((x, y, x + relative_patch_size, y + relative_patch_size))

                _trial = 0
                while self.get_tight_bound_size(np.array(crop_candidate)) < relative_patch_size // 16 and _trial < MAX_TRIAL:
                    x = np.random.randint(0, offset)
                    y = np.random.randint(0, offset)
                    crop_candidate = image.crop((x, y, x + relative_patch_size, y + relative_patch_size))
                    _trial += 1

                crop_candidate = np.array(crop_candidate.resize((self.args.imsize, self.args.imsize)))
            image_list.append(crop_candidate)
        return image_list

    def get_pairs(self, content_english=False, style_english=False):
        lang_content = random.choice(self.lang_list)

        content_unicode_list = english_set if content_english else self.data[lang_content]
        style_unicode_list = english_set if style_english else self.data[lang_content]

        if content_english == style_english:
            # content_unicode_list == style_unicode_list
            chars = random.sample(content_unicode_list,
                                  k=self.args.reference_imgs.style + 1)
            content_char = chars[-1]
            style_chars = chars[:self.args.reference_imgs.style]
        else:
            content_char = random.choice(content_unicode_list)
            style_chars = random.sample(style_unicode_list, k=self.args.reference_imgs.style)

        # fonts = random.sample(self.content_meta[lang_content].keys(),
        #                       k=self.args.reference_imgs.char + 1)
        # content_fonts = fonts[:self.args.reference_imgs.char]
        # style_font = fonts[-1]
        
        style_font_list = list(self.content_meta[lang_content].keys())
        style_font_list.remove(NOTO_FONT_DIRNAME)
        style_font = random.choice(style_font_list)
        content_fonts = [NOTO_FONT_DIRNAME]

        content_fonts_image = [self.content_meta[lang_content][x][content_char] for x in content_fonts]
        style_chars_image = [self.content_meta[lang_content][style_font][x] for x in style_chars]

        # style_chars_image = [self.content_meta[lang_content][style_font][x] for x in style_chars]

        # style_chars_cropped = []
        # for style_char_image in style_chars_image:
        #     style_chars_cropped.extend(self.get_patch_from_style_image(style_char_image,
        #                                                                patch_per_image=self.args.reference_imgs.style // self.args.reference_imgs.char))

        target_image = self.content_meta[lang_content][style_font][content_char]
        
        content_fonts_image = [self._get_content_image(image_path) for image_path in content_fonts_image]
        style_chars_image = [self._get_content_image(image_path) for image_path in style_chars_image]
        target_image = self._get_content_image(target_image)

        return content_char, content_fonts, content_fonts_image, style_font, style_chars, style_chars_image, target_image

    def __getitem__(self, idx):
        """GoogleFontDataset의 __getitem__

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
        use_eng_content, use_eng_style = random.choice([(True, False), (False, True), (False, False)])

        if self.mode != 'train':
            use_eng_content = False
            use_eng_style = True

        content_char, content_fonts, content_fonts_image, style_font, style_chars, style_chars_image, target_image = \
            self.get_pairs(content_english=use_eng_content, style_english=use_eng_style)

        content_fonts_image = np.array([np.mean(np.array(x), axis=-1) / WHITE
                                        for x in content_fonts_image], dtype=np.float32)
        style_chars_image = np.array([np.mean(np.array(x), axis=-1) / WHITE
                                      for x in style_chars_image], dtype=np.float32)
        target_image = np.mean(np.array(target_image,  dtype=np.float32), axis=-1)[np.newaxis, ...] / WHITE

        dict_return = {
            # data for training
            'gt_images': target_image,
            'content_images': content_fonts_image,
            'style_images': style_chars_image,  # TODO: crop style image with fixed size
            # data for logging
            'style_idx': style_font,
            'char_idx': content_char,
            'content_image_idxs': content_fonts,
            'style_image_idxs': style_chars,
            'image_paths': '',
        }
        return dict_return

    def __len__(self):
        return len(self.lang_list) * REPEATE_NUM


if __name__ == '__main__':
    hp = OmegaConf.load('config/datasets/googlefont.yaml').datasets.train
    metadata_path = "./lang_set.json"
    FONT_DIR = "/data2/hksong/DATA/fonts-image"

    _dataset = GoogleFontDataset(hp, metadata_path=metadata_path, font_dir=FONT_DIR)
    TEST_ITER_NUM = 4
    for i in range(TEST_ITER_NUM):
        data = _dataset[i]
        print(data.keys())
        print(data['gt_image'].size,
              data['content_images'][0].size,
              data['style_images'][0].size,
              data['lang'],
              data['style_idx'],
              data['char_idx'],
              data['content_image_idxs'],
              data['style_image_idxs'])
