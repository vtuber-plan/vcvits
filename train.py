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

def old():
    
    if rank == 0:
        eval_dataset = TextAudioLoader(hparams.data.validation_files, hparams.data)
        eval_loader = DataLoader(eval_dataset, num_workers=8, shuffle=False,
            batch_size=hparams.train.batch_size, pin_memory=True,
            drop_last=False, collate_fn=collate_fn)

    
    optim_g = torch.optim.AdamW(
        net_g.parameters(), 
        hps.train.learning_rate, 
        betas=hps.train.betas, 
        eps=hps.train.eps)
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        hps.train.learning_rate, 
        betas=hps.train.betas, 
        eps=hps.train.eps)
    net_g = DDP(net_g, device_ids=[rank])
    net_d = DDP(net_d, device_ids=[rank])

    try:
        _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g, optim_g)
        _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"), net_d, optim_d)
        global_step = (epoch_str - 1) * len(train_loader)
    except:
        epoch_str = 1
        global_step = 0

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str-2)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str-2)

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
