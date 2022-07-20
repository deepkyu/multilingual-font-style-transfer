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
    parser = argparse.ArgumentParser(description='Code to train font style transfer')

    parser.add_argument("--config", type=str, default="./config/setting.yaml",
                        help="Config file for training")
    parser.add_argument('-g', '--gpus', type=str, default=None,
                        help="Number of gpus to use (e.g. '0,1,2,3'). Will use all if not given.")
    parser.add_argument('-p', '--resume_checkpoint_path', type=str, default=None,
                        help="path of checkpoint for resuming")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    hp, pl_config = load_configuration(args.config)
    
    logging_dir = Path(hp.logging.log_dir)

    # call lightning module
    font_pl = FontLightningModule(hp)

    # set logging
    hp.logging['log_dir'] = logging_dir / 'tensorboard'
    savefiles = []
    for reg in hp.logging.savefiles:
        savefiles += glob.glob(reg)
    hp.logging['log_dir'].mkdir(exist_ok=True)
    save_files(str(logging_dir), savefiles)

    # set tensorboard logger
    logger = TensorBoardLogger(str(logging_dir), name=str(hp.logging.seed))
    
    # set checkpoing callback
    weights_save_path = logging_dir / 'checkpoint'
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(weights_save_path),
        **pl_config.checkpoint.callback
    )

    # set lightning trainer
    trainer = pl.Trainer(
        logger=logger,
        gpus=-1 if args.gpus is None else args.gpus,
        callbacks=[checkpoint_callback],
        weights_save_path=weights_save_path,
        resume_from_checkpoint=args.resume_checkpoint_path,
        **pl_config.trainer
    )
    
    # let's train
    trainer.fit(font_pl)

if __name__ == "__main__":
    main()
