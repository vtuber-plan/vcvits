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
from vits.preprocess import preprocess
from vits.model.preload_vcvits import PreloadVCVITS

import vits.utils as utils
from vits.data.collate import PreloadAnyVoiceConversionMultiSpeakerCollate

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from vits.hparams import HParams
from vits.data.dataset.preload_vc_ms import PreloadAnyVoiceConversionMultiSpeakerDataset

def get_hparams(config_path: str) -> HParams:
    with open(config_path, "r") as f:
        data = f.read()
    config = json.loads(data)
    
    hparams = HParams(**config)
    return hparams

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default="./configs/base.json", help='JSON file for configuration')
    parser.add_argument('-a', '--accelerator', type=str, default="gpu", help='training device')
    parser.add_argument('-d', '--device', type=str, default="0", help='training device ids')
    parser.add_argument('-s', '--skip-preprocess', action='store_true', help='Skip preprocess')
    args = parser.parse_args()

    hparams = get_hparams(args.config)
    pl.utilities.seed.seed_everything(hparams.train.seed)

    train_dataset = PreloadAnyVoiceConversionMultiSpeakerDataset(hparams.data.training_files, hparams.data)
    valid_dataset = PreloadAnyVoiceConversionMultiSpeakerDataset(hparams.data.validation_files, hparams.data)

    if "skip-preprocess" not in args:
        preprocess(hparams, hparams.data.training_files, hparams.data.source_sampling_rate, load_features=True)
        preprocess(hparams, hparams.data.training_files, hparams.data.target_sampling_rate)
        preprocess(hparams, hparams.data.validation_files, hparams.data.source_sampling_rate, load_features=True)
        preprocess(hparams, hparams.data.validation_files, hparams.data.target_sampling_rate)
    
    collate_fn = PreloadAnyVoiceConversionMultiSpeakerCollate()
    train_loader = DataLoader(train_dataset, batch_size=hparams.train.batch_size, num_workers=16, shuffle=True, pin_memory=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=1, num_workers=16, shuffle=False, pin_memory=True, collate_fn=collate_fn)

    model = PreloadVCVITS(**hparams)

    checkpoint_callback = ModelCheckpoint(dirpath=None, save_last=True, every_n_train_steps=500)

    devices = [int(n.strip()) for n in args.device.split(",")]
    trainer_params = {
        "accelerator": args.accelerator,
        "devices": devices,
        
        "callbacks": [checkpoint_callback],
    }

    if len(devices) > 1:
        trainer_params["strategy"] = "ddp"

    trainer_params.update(hparams.trainer)

    if hparams.train.fp16_run:
        trainer_params["amp_backend"] = "native"
        trainer_params["precision"] = 16
    
    trainer = pl.Trainer(**trainer_params)
    # resume training
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
