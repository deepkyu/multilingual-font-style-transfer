import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import linalg
import torch
import torchvision.transforms.functional as TF

from trainer.custom_model import CustomModel
from .ssim import SSIM, MSSSIM
from .fid import FID
from .classifier import Classifier


class Evaluator():
    def __init__(self, args):
        self.args = args

        self.device = self.args.logging.device
        try:
            shutil.rmtree(self.args.logging.results_dir)
        except:
            pass
        os.makedirs(self.args.logging.results_dir, exist_ok=True)
        self.args.logging.result_file = os.path.join(self.args.logging.results_dir, 'results.txt')
        self.args.logging.final_result_file = os.path.join(self.args.logging.results_dir, 'final_results.txt')
        if os.path.exists(self.args.logging.result_file):
            os.remove(self.args.logging.result_file)

        self.model = CustomModel(args)
        self.model.eval()

        self.criterion = {}
        self.criterion['L1'] = torch.nn.L1Loss()
        self.criterion['SSIM'] = SSIM(val_range=1).to(self.device)  # img value is in [0, 1]
        self.criterion['MSSSIM'] = MSSSIM(weights=[0.45, 0.3, 0.25], val_range=1).to(
            self.device)  # since imsize=64, len(weight)<=3
        self.criterion['FID'] = FID()

        self.results = {}
        self.final_results = {}
        self.fid_values = {}

        self.steps = 0
        self.epochs = 0

    def process_image(self, value):
        assert value.ndim == 4, value.shape
        # B x C x H x W -> B x H x (CxW)
        value = torch.cat([value[:, j] for j in range(value.shape[1])], dim=-1)
        value = value.unsqueeze(1)  # B x 1 x H x (CxW)
        return value

    def evaluate(self, data):
        self.results = {}

        self.model.set_input(data)
        self.model.forward()
        self.model.calc_metrics()

        path_imsave = os.path.join(self.args.logging.results_dir, 'images')
        path_gen = os.path.join(path_imsave, 'gen')
        path_gt = os.path.join(path_imsave, 'gt')
        path_all = os.path.join(path_imsave, 'all')
        os.makedirs(path_imsave, exist_ok=True)
        os.makedirs(path_gen, exist_ok=True)
        os.makedirs(path_gt, exist_ok=True)
        os.makedirs(path_all, exist_ok=True)

        if self.args.logging.save_images:
            # log imgs, logs
            img_save = []
            for key, value in self.model.log.items():
                if isinstance(value, torch.Tensor) and 'image' in key:
                    img_save.append(self.process_image(value))
            img_save = torch.cat(img_save, dim=-1)  # B x 1 x H x (SUM CxW)
            img_save = torch.cat([img_save[i] for i in range(len(img_save))], dim=-2)  # 1 x (BxH) x (W')
            path_save = os.path.join(path_all, f'{self.epochs}_{self.steps}.png')
            if img_save.shape[0] == 1:
                img_save = torch.repeat_interleave(img_save, 3, dim=0)
            img_save = TF.to_pil_image(img_save.detach().cpu())
            img_save.save(path_save)

            gt_im = self.process_image(data['gt_images'])
            gt_im = torch.cat([gt_im[i] for i in range(len(gt_im))], dim=-2)
            path_save = os.path.join(path_gt, f'{self.epochs}_{self.steps}.png')
            if gt_im.shape[0] == 1:
                gt_im = torch.repeat_interleave(gt_im, 3, dim=0)
            gt_im = TF.to_pil_image(gt_im.detach().cpu())
            gt_im.save(path_save)

            gen_im = self.process_image(self.model.log['generated_images'])
            gen_im = torch.cat([gen_im[i] for i in range(len(gen_im))], dim=-2)
            path_save = os.path.join(path_gen, f'{self.epochs}_{self.steps}.png')
            if gen_im.shape[0] == 1:
                gen_im = torch.repeat_interleave(gen_im, 3, dim=0)
            gen_im = TF.to_pil_image(gen_im.detach().cpu())
            gen_im.save(path_save)

        data.update(self.model.log)
        self.results['batch_size'] = len(data['style_idx'])
        self.results['style_idx'] = data['style_idx']
        self.results['char_idx'] = data['char_idx']

        self.compute_l1(data['gt_images'], data['generated_images'])
        self.compute_ssim(data['gt_images'], data['generated_images'])
        self.compute_msssim(data['gt_images'], data['generated_images'])

        for key, model in self.model.models.items():
            if key == 'classify_char':
                self.compute_acc(model, data['generated_images'], data['char_idx'], 'char')
                # self.compute_fid(model, data['gt_images'], data['generated_images'], 'char')
            elif key == 'classify_style':
                self.compute_acc(model, data['generated_images'], data['style_idx'], 'style')
                # self.compute_fid(model, data['gt_images'], data['generated_images'], 'style')

    def compute_l1(self, gt, gen):
        keys = self.model.log.keys()
        for key in ['G_L1', 'L1']:
            if key in keys:
                self.results['L1'] = self.model.log[key]
                return

        self.results['L1'] = self.criterion['L1'](gt, gen).item()

    def compute_ssim(self, gt, gen):
        keys = self.model.log.keys()
        for key in ['SSIM']:
            if key in keys:
                self.results['SSIM'] = self.model.log[key]
                return

        self.results['ssim'] = self.criterion['SSIM'](gt, gen).item()

    def compute_msssim(self, gt, gen):
        keys = self.model.log.keys()
        for key in ['MSSSIM']:
            if key in keys:
                self.results['MSSSIM'] = self.model.log[key]
                return

        msssim_value = self.criterion['MSSSIM'](gt, gen).item()
        if np.isnan(msssim_value):
            msssim_value = 0.
        self.results['msssim'] = msssim_value

    def compute_acc(self, classifier, imgs, labels, key):
        with torch.no_grad():
            imgs = torch.repeat_interleave(imgs, 3, dim=1)

            feat = classifier(imgs)[0]
            while feat.ndim > 2:
                feat = feat.squeeze(-1)
            predicts = torch.argmax(feat, dim=-1, keepdim=True)
        predicts = predicts.view(-1, 1).detach().cpu()
        labels = labels.view(-1, 1).detach().cpu()
        self.results[f'acc_{key}'] = torch.sum(predicts == labels) / len(labels)

    def compute_fid(self, model, gt, gen, key):
        raise NotImplementedError

    ###############################################

    def record_current_results(self, verbose=False):
        res = []
        res += ['################################\n']

        for key, value in self.results.items():
            res += [f'{key}:{value}\n']

            if 'idx' not in key:
                if key in self.final_results.keys():
                    self.final_results[key] += value
                else:
                    self.final_results[key] = value

        if verbose:
            print('----------- current results -------------')
            print(res)

        with open(self.args.logging.result_file, 'a') as f:
            f.writelines(res)

    def compute_final_results(self, verbose=False):
        if 'classify_char' in self.model.models.keys():
            self.compute_final_fid('char')
        if 'classify_style' in self.model.models.keys():
            self.compute_final_fid('style')

        results = {}
        for key, value in self.final_results.items():
            results[key] = float(value / self.steps)

        res = [f'{key}:{value}\n' for key, value in results.items()]
        if verbose:
            print(res)
        with open(self.args.logging.final_result_file, 'w') as f:
            f.writelines(res)
        print(f'results saved at {self.args.logging.final_result_file}')
