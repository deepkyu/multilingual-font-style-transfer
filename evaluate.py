import argparse
import glob
from pathlib import Path

from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from lightning import FontLightningModule
from utils import save_files


def load_configuration(path_config):
    setting = OmegaConf.load(path_config)

    # load hyperparameter
    hp = OmegaConf.load(setting.config.dataset)
    hp = OmegaConf.merge(hp, OmegaConf.load(setting.config.model))
    hp = OmegaConf.merge(hp, OmegaConf.load(setting.config.logging))

    # with lightning setting
    if hasattr(setting.config, 'lightning'):
        pl_config = OmegaConf.load(setting.config.lightning)
        if hasattr(pl_config, 'pl_config'):
            return hp, pl_config.pl_config
        return hp, pl_config

    # without lightning setting
    return hp


def parse_args():
    parser = argparse.ArgumentParser(description='Code to evaluate font style transfer')

    parser.add_argument('-c', "--config", type=str, default="./config/setting.yaml",
                        help="Config file for evaluation")
    parser.add_argument('-g', '--gpus', type=str, default='0,1',
                        help="Number of gpus to use (e.g. '0,1,2,3'). Will use all if not given.")
    parser.add_argument('-p', '--resume_checkpoint_path', type=str, required=True,
                        help="path of checkpoint for evaluation")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    hp, pl_config = load_configuration(args.config)

    # call lightning module
    font_pl = FontLightningModule(hp)

    # set lightning trainer
    trainer = pl.Trainer(
        gpus=-1 if args.gpus is None else args.gpus,
    )

    # let's train
    trainer.test(model=font_pl, ckpt_path=args.resume_checkpoint_path)


if __name__ == "__main__":
    main()
