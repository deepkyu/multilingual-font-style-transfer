import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import importlib
import PIL.Image as Image

import models
import datasets
from evaluator.ssim import SSIM, MSSSIM
import lpips
from models.loss import GANHingeLoss
from utils import set_logger, magic_image_handler

NUM_TEST_SAVE_IMAGE = 10


class FontLightningModule(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.losses = {}
        self.metrics = {}
        self.networks = nn.ModuleDict(self.build_models())
        self.module_keys = list(self.networks.keys())

        self.losses = self.build_losses()
        self.metrics = self.build_metrics()

        self.opt_tag = {key: None for key in self.networks.keys()}
        self.sched_tag = {key: None for key in self.networks.keys()}
        self.sched_use = False
        # self.automatic_optimization = False

        self.train_d_content = True
        self.train_d_style = True

    def build_models(self):
        networks = {}
        for key, hp_model in self.args.models.items():
            key_ = key.lower()
            if 'g' == key_[0]:
                model_ = models.Generator(hp_model)
            elif 'd' == key_[0]:
                model_ = models.PatchGANDiscriminator(hp_model)  # TODO: add option for selecting discriminator
            else:
                raise ValueError(f"No key such as {key}")

            networks[key.lower()] = model_
        return networks

    def build_losses(self):
        losses_dict = {}
        losses_dict['L1'] = torch.nn.L1Loss()

        if 'd_content' in self.module_keys:
            losses_dict['GANLoss_content'] = GANHingeLoss()
        if 'd_style' in self.module_keys:
            losses_dict['GANLoss_style'] = GANHingeLoss()

        return losses_dict

    def build_metrics(self):
        metrics_dict = nn.ModuleDict()
        metrics_dict['ssim'] = SSIM(val_range=1)  # img value is in [0, 1]
        metrics_dict['msssim'] = MSSSIM(weights=[0.45, 0.3, 0.25], val_range=1)  # since imsize=64, len(weight)<=3
        metrics_dict['lpips'] = lpips.LPIPS(net='vgg')
        return metrics_dict

    def configure_optimizers(self):
        optims = {}
        for key, args_model in self.args.models.items():
            key = key.lower()
            if args_model['optim'] is not None:
                args_optim = args_model['optim']
                module, cls = args_optim['class'].rsplit(".", 1)
                O = getattr(importlib.import_module(module, package=None), cls)
                o = O([p for p in self.networks[key].parameters() if p.requires_grad],
                      lr=args_optim.lr, betas=args_optim.betas)

                optims[key] = o

        optim_module_keys = optims.keys()

        count = 0
        optim_list = []

        for _key in self.module_keys:
            if _key in optim_module_keys:
                optim_list.append(optims[_key])
                self.opt_tag[_key] = count
                count += 1

        return optim_list

    def forward(self, content_images, style_images):
        return self.networks['g']((content_images, style_images))

    def common_forward(self, batch, batch_idx):
        loss = {}
        logs = {}

        content_images = batch['content_images']
        style_images = batch['style_images']
        gt_images = batch['gt_images']
        image_paths = batch['image_paths']
        char_idx = batch['char_idx']

        generated_images = self(content_images, style_images)

        # l1 loss
        loss['g_L1'] = self.losses['L1'](generated_images, gt_images)
        loss['g_backward'] = loss['g_L1'] * self.args.logging.lambda_L1

        # loss for training generator
        if 'd_content' in self.module_keys:
            loss = self.d_content_loss_for_G(content_images, generated_images, loss)

        if 'd_style' in self.networks.keys():
            loss = self.d_style_loss_for_G(style_images, generated_images, loss)

        # loss for training discriminator
        generated_images = generated_images.detach()

        if 'd_content' in self.module_keys:
            if self.train_d_content:
                loss = self.d_content_loss_for_D(content_images, generated_images, gt_images, loss)

        if 'd_style' in self.module_keys:
            if self.train_d_style:
                loss = self.d_style_loss_for_D(style_images, generated_images, gt_images, loss)

        logs['content_images'] = content_images
        logs['style_images'] = style_images
        logs['gt_images'] = gt_images
        logs['generated_images'] = generated_images

        return loss, logs

    @property
    def automatic_optimization(self):
        return False

    def training_step(self, batch, batch_idx):
        metrics = {}
        # forward
        loss, logs = self.common_forward(batch, batch_idx)

        if self.global_step % self.args.logging.freq['train'] == 0:
            with torch.no_grad():
                metrics.update(self.calc_metrics(logs['gt_images'], logs['generated_images']))

        # backward
        opts = self.optimizers()

        opts[self.opt_tag['g']].zero_grad()
        self.manual_backward(loss['g_backward'])

        if 'd_content' in self.module_keys:
            if self.train_d_content:
                opts[self.opt_tag['d_content']].zero_grad()
                self.manual_backward(loss['dcontent_backward'])

        if 'd_style' in self.module_keys:
            if self.train_d_style:
                opts[self.opt_tag['d_style']].zero_grad()
                self.manual_backward(loss['dstyle_backward'])

        opts[self.opt_tag['g']].step()

        if 'd_content' in self.module_keys:
            if self.train_d_content:
                opts[self.opt_tag['d_content']].step()

        if 'd_style' in self.module_keys:
            if self.train_d_style:
                opts[self.opt_tag['d_style']].step()

        if self.global_step % self.args.logging.freq['train'] == 0:
            self.custom_log(loss, metrics, logs, mode='train')

    def validation_step(self, batch, batch_idx):
        metrics = {}
        loss, logs = self.common_forward(batch, batch_idx)
        self.custom_log(loss, metrics, logs, mode='eval')

    def test_step(self, batch, batch_idx):
        metrics = {}
        loss, logs = self.common_forward(batch, batch_idx)
        metrics.update(self.calc_metrics(logs['gt_images'], logs['generated_images']))

        if batch_idx < NUM_TEST_SAVE_IMAGE:
            for key, value in logs.items():
                if 'image' in key:
                    sample_images = (magic_image_handler(value) * 255)[..., 0].astype(np.uint8)
                    Image.fromarray(sample_images).save(f"{batch_idx:02d}_{key}.png")

        return loss, logs, metrics

    def test_epoch_end(self, test_step_outputs):
        # do something with the outputs of all test batches
        # all_test_preds = test_step_outputs.metrics
        ssim_list = []
        msssim_list = []

        for _, test_output in enumerate(test_step_outputs):

            ssim_list.append(test_output[2]['SSIM'].cpu().numpy())
            msssim_list.append(test_output[2]['MSSSIM'].cpu().numpy())

        print(f"SSIM: {np.mean(ssim_list)}")
        print(f"MSSSIM: {np.mean(msssim_list)}")

    def common_dataloader(self, mode='train', batch_size=None):
        dataset_cls = getattr(datasets, self.args.datasets.type)
        dataset_config = getattr(self.args.datasets, mode)
        dataset = dataset_cls(dataset_config, mode=mode)
        _batch_size = batch_size if batch_size is not None else dataset_config.batch_size
        dataloader = DataLoader(dataset,
                                shuffle=dataset_config.shuffle,
                                batch_size=_batch_size,
                                num_workers=dataset_config.num_workers,
                                drop_last=True)

        return dataloader

    def train_dataloader(self):
        return self.common_dataloader(mode='train')

    def val_dataloader(self):
        return self.common_dataloader(mode='eval')

    def test_dataloader(self):
        return self.common_dataloader(mode='train')

    def calc_metrics(self, gt_images, generated_images):
        """

        :param gt_images:
        :param generated_images:
        :return:
        """
        metrics = {}
        _gt = torch.clamp(gt_images.clone(), 0, 1)
        _gen = torch.clamp(generated_images.clone(), 0, 1)
        metrics['SSIM'] = self.metrics['ssim'](_gt, _gen)
        msssim_value = self.metrics['msssim'](_gt, _gen)
        metrics['MSSSIM'] = msssim_value if not torch.isnan(msssim_value) else torch.tensor(0.).type_as(_gt)
        metrics['LPIPS'] = self.metrics['lpips'](_gt * 2 - 1, _gen * 2 - 1).squeeze().mean()
        return metrics

    # region step
    def d_content_loss_for_G(self, content_images, generated_images, loss):
        pred_generated = self.networks['d_content'](torch.cat([content_images, generated_images], dim=1))
        loss['g_gan_content'] = self.losses['GANLoss_content'](pred_generated, True, for_discriminator=False)

        loss['g_backward'] += loss['g_gan_content']
        return loss

    def d_content_loss_for_D(self, content_images, generated_images, gt_images, loss):
        # D
        if 'd_content' in self.module_keys:
            if self.train_d_content:
                pred_gt_images = self.networks['d_content'](torch.cat([content_images, gt_images], dim=1))
                pred_generated_images = self.networks['d_content'](torch.cat([content_images, generated_images], dim=1))

                loss['dcontent_gt'] = self.losses['GANLoss_content'](pred_gt_images, True, for_discriminator=True)
                loss['dcontent_gen'] = self.losses['GANLoss_content'](pred_generated_images, False, for_discriminator=True)
                loss['dcontent_backward'] = (loss['dcontent_gt'] + loss['dcontent_gen'])

        return loss

    def d_style_loss_for_G(self, style_images, generated_images, loss):
        pred_generated = self.networks['d_style'](torch.cat([style_images, generated_images], dim=1))
        loss['g_gan_style'] = self.losses['GANLoss_style'](pred_generated, True, for_discriminator=False)

        assert self.train_d_style
        loss['g_backward'] += loss['g_gan_style']
        return loss

    def d_style_loss_for_D(self, style_images, generated_images, gt_images, loss):
        pred_gt_images = self.networks['d_style'](torch.cat([style_images, gt_images], dim=1))
        pred_generated_images = self.networks['d_style'](torch.cat([style_images, generated_images], dim=1))

        loss['dstyle_gt'] = self.losses['GANLoss_style'](pred_gt_images, True, for_discriminator=True)
        loss['dstyle_gen'] = self.losses['GANLoss_style'](pred_generated_images, False, for_discriminator=True)
        loss['dstyle_backward'] = (loss['dstyle_gt'] + loss['dstyle_gen'])

        return loss

    def custom_log(self, loss, metrics, logs, mode):
        # logging values with tensorboard
        for loss_full_key, value in loss.items():
            model_type, loss_type = loss_full_key.split('_')[0], "_".join(loss_full_key.split('_')[1:])
            self.log(f'{model_type}/{mode}_{loss_type}', value)

        for metric_full_key, value in metrics.items():
            model_type, metric_type = metric_full_key.split('_')[0], "_".join(metric_full_key.split('_')[1:])
            self.log(f'{model_type}/{mode}_{metric_type}', value)

        # logging images, params, etc.
        tensorboard = self.logger.experiment
        for key, value in logs.items():
            if 'image' in key:
                sample_images = magic_image_handler(value)
                tensorboard.add_image(f"{mode}/" + key, sample_images, self.global_step, dataformats='HWC')
            elif 'param' in key:
                tensorboard.add_histogram(f"{mode}" + key, value, self.global_step)
            else:
                raise RuntimeError(f"Only logging with one of keywords: image, param | current input: {key}")
