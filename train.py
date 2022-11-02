import os
import json
import glob
import argparse
import torch
import torchaudio
import fairseq
import tqdm
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from preprocess import preprocess
from vits.model.preload_vcvits import PreloadVCVITS

import vits.utils as utils
from vits.data.collate import PreloadAnyVoiceConversionMultiSpeakerCollate

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from vits.hparams import HParams
from vits.data.dataset.preload_vc_ms import PreloadAnyVoiceConversionMultiSpeakerLoader

def get_hparams() -> HParams:
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default="./configs/base.json", help='JSON file for configuration')
    args = parser.parse_args()

    with open(args.config, "r") as f:
        data = f.read()
    config = json.loads(data)
    
    hparams = HParams(**config)
    return hparams

def main():
    hparams = get_hparams()
    pl.utilities.seed.seed_everything(hparams.train.seed)

    train_dataset = PreloadAnyVoiceConversionMultiSpeakerLoader(hparams.data.training_files, hparams.data)
    valid_dataset = PreloadAnyVoiceConversionMultiSpeakerLoader(hparams.data.validation_files, hparams.data)

    # preprocess(hparams, hparams.data.training_files, hparams.data.source_sampling_rate, load_features=True)
    # preprocess(hparams, hparams.data.training_files, hparams.data.target_sampling_rate)
    # preprocess(hparams, hparams.data.validation_files, hparams.data.source_sampling_rate, load_features=True)
    # preprocess(hparams, hparams.data.validation_files, hparams.data.target_sampling_rate)
    
    collate_fn = PreloadAnyVoiceConversionMultiSpeakerCollate()
    train_loader = DataLoader(train_dataset, batch_size=hparams.train.batch_size, num_workers=16, shuffle=False, pin_memory=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=1, num_workers=16, shuffle=False, pin_memory=True, collate_fn=collate_fn)

    model = PreloadVCVITS(**hparams)

    checkpoint_callback = ModelCheckpoint(dirpath=None, save_last=True, every_n_train_steps=500)

    trainer_params = {
        "accelerator": "gpu",
        "devices": [0, 1, 2, 3],
        "strategy": "ddp",
        "callbacks": [checkpoint_callback],
    }

    trainer_params.update(hparams.trainer)

    if hparams.train.fp16_run:
        trainer_params["amp_backend"] = "native"
        trainer_params["precision"] = 16
    
    trainer = pl.Trainer(**trainer_params)
    # recover training
    ckpt_path = None
    if os.path.exists("logs/lightning_logs"):
        versions = glob.glob("logs/lightning_logs/version_*")
        if len(list(versions)) > 0:
            last_ver = sorted(list(versions))[-1]
            last_ckpt = os.path.join(last_ver, "checkpoints/last.ckpt")
            if os.path.exists(last_ckpt):
                ckpt_path = last_ckpt
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader, ckpt_path=ckpt_path)

if __name__ == "__main__":
  main()
