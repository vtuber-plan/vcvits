import os
import json
import argparse
import itertools
import math
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

import vits.utils as utils
from vits.data.collate import (
  TextAudioCollate,
)

import pytorch_lightning as pl

from vits.hparams import HParams
from vits.model.vits import VITS
from vits.data.dataset import TextAudioLoader, TextAudioSpeakerLoader

def get_hparams() -> HParams:
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default="./configs/paimoon_base.json", help='JSON file for configuration')
    # parser.add_argument('-m', '--model', type=str, required=True, help='Model name')
    args = parser.parse_args()

    with open(args.config, "r") as f:
        data = f.read()
    config = json.loads(data)
    
    hparams = HParams(**config)
    return hparams

def main():
    hparams = get_hparams()
    train_dataset = TextAudioLoader(hparams.data.training_files, hparams.data)
    valid_dataset = TextAudioLoader(hparams.data.validation_files, hparams.data)

    collate_fn = TextAudioCollate()
    train_loader = DataLoader(train_dataset, batch_size=hparams.train.batch_size, num_workers=16, shuffle=False, pin_memory=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=1, num_workers=16, shuffle=False, pin_memory=True, collate_fn=collate_fn)

    model = VITS(**hparams)
    
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[3],
        # strategy="ddp",
        # amp_backend="native",
        # precision=16,
        # logger=logger,
        # max_steps=100,
        accumulate_grad_batches=8,
        max_epochs=hparams.train.epochs,
        default_root_dir="./logs",
        limit_val_batches=1
    )

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader)

if __name__ == "__main__":
  main()
