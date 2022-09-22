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

def get_hparams():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default="./configs/base.json", help='JSON file for configuration')
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

    collate_fn = TextAudioCollate()
    train_loader = DataLoader(train_dataset, num_workers=8, shuffle=False, pin_memory=True, collate_fn=collate_fn)

    model = VITS(hparams)
    
    trainer = pl.Trainer(
        accelerator="cpu",
        # devices=[0],
        # logger=logger,
        # max_steps=100,
        max_epochs=100,
        default_root_dir="./logs",
    )

    trainer.fit(model=model, train_dataloaders=train_loader)

if __name__ == "__main__":
  main()
