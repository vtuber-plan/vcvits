import os
import json
import glob
import argparse
from typing import Optional
import torch
import torchaudio
import fairseq
import tqdm
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from vits.preprocess import preprocess
from vits.light.vcvits import VCVITS

import vits.utils as utils
from vits.data.collate import VoiceConversionMultiSpeakerCollate

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.profilers import SimpleProfiler, AdvancedProfiler

import lightning_fabric

from vits.hparams import HParams
from vits.data.dataset.vc_ms import VoiceConversionMultiSpeakerDataset

from joblib import Parallel, delayed
import warnings

def get_hparams(config_path: str) -> HParams:
    with open(config_path, "r") as f:
        data = f.read()
    config = json.loads(data)
    
    hparams = HParams(**config)
    return hparams

def last_checkpoint(path: str) -> Optional[str]:
    ckpt_path = None
    if os.path.exists(os.path.join(path, "lightning_logs")):
        versions = glob.glob(os.path.join(path, "lightning_logs", "version_*"))
        if len(list(versions)) > 0:
            last_ver = sorted(list(versions), key=lambda p: int(p.split("_")[-1]))[-1]
            last_ckpt = os.path.join(last_ver, "checkpoints/last.ckpt")
            if os.path.exists(last_ckpt):
                ckpt_path = last_ckpt
    return ckpt_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default="./configs/48k_base.json", help='JSON file for configuration')
    parser.add_argument('-a', '--accelerator', type=str, default="cpu", help='training device')
    parser.add_argument('-d', '--device', type=str, default="0", help='training device ids')
    parser.add_argument('-s', '--skip-preprocess', action='store_true', help='Skip preprocess')
    parser.add_argument('-cd', '--cachedir', type=str, default="./dataset_cache", help='Dataset cache')
    args = parser.parse_args()

    hparams = get_hparams(args.config)
    lightning_fabric.utilities.seed.seed_everything(hparams.train.seed)
    cache_dir = args.cachedir if len(args.cachedir.strip()) != 0 else None

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)

    train_dataset = VoiceConversionMultiSpeakerDataset(hparams.data.training_files, hparams.data, cache_dir)
    valid_dataset = VoiceConversionMultiSpeakerDataset(hparams.data.validation_files, hparams.data, cache_dir)

    # preprocess
    print("Preprocess...")
    if args.skip_preprocess:
        print("skip preprocessing..")
    else:
        Parallel(n_jobs=64, backend="loky")\
            (delayed(train_dataset.get_item)(i, 0) \
                for i in tqdm.tqdm(range(len(train_dataset))))

        for data in tqdm.tqdm(valid_dataset):
            pass
        
    collate_fn = VoiceConversionMultiSpeakerCollate()
    train_loader = DataLoader(train_dataset, batch_size=hparams.train.batch_size, num_workers=16, shuffle=True, pin_memory=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=1, num_workers=16, shuffle=False, pin_memory=True, collate_fn=collate_fn)

    model = VCVITS(**hparams)
    # model = VCVITS.load_from_checkpoint(checkpoint_path="logs/lightning_logs/ver_1/checkpoints/last.ckpt", strict=False)

    checkpoint_callback = ModelCheckpoint(dirpath=None, save_last=True, every_n_train_steps=2000)

    devices = [int(n.strip()) for n in args.device.split(",")]
    trainer_params = {
        "accelerator": args.accelerator,
        "callbacks": [checkpoint_callback],
    }

    if args.accelerator != "cpu":
        trainer_params["devices"] = devices

    if len(devices) > 1:
        trainer_params["strategy"] = "ddp"

    trainer_params.update(hparams.trainer)

    if hparams.train.fp16_run:
        trainer_params["amp_backend"] = "native"
        trainer_params["precision"] = 16
    
    # profiler = AdvancedProfiler(filename="profile.txt")
    
    trainer = pl.Trainer(**trainer_params) # , profiler=profiler, max_steps=200
    # resume training
    ckpt_path = last_checkpoint(hparams.trainer.default_root_dir)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader, ckpt_path=ckpt_path)

if __name__ == "__main__":
  main()
