import os
import json
import argparse
import itertools
import math
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler

import commons
import utils
from data_utils import (
  TextAudioLoader,
  TextAudioCollate,
  DistributedBucketSampler
)

from losses import (
  generator_loss,
  discriminator_loss,
  feature_loss,
  kl_loss
)
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from text.symbols import symbols


import pytorch_lightning as pl

from vits.hparams import HParams

def get_hparams():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default="./configs/base.json", help='JSON file for configuration')
    parser.add_argument('-m', '--model', type=str, required=True, help='Model name')
    args = parser.parse_args()

    with open(args.config, "r") as f:
        data = f.read()
    config = json.loads(data)
    
    hparams = HParams(**config)
    return hparams

def main():
    hparams = get_hparams()
    train_dataset = TextAudioLoader(hparams.data.training_files, hparams.data)
    train_sampler = DistributedBucketSampler(
        train_dataset,
        hparams.train.batch_size,
        [32,300,400,500,600,700,800,900,1000],
        num_replicas=n_gpus,
        rank=rank,
        shuffle=True)
    collate_fn = TextAudioCollate()
    train_loader = DataLoader(train_dataset, num_workers=8, shuffle=False, pin_memory=True, collate_fn=collate_fn, batch_sampler=train_sampler)
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

    scaler = GradScaler(enabled=hps.train.fp16_run)

    for epoch in range(epoch_str, hps.train.epochs + 1):
        if rank==0:
        train_and_evaluate(rank, epoch, hps, [net_g, net_d], [optim_g, optim_d], [scheduler_g, scheduler_d], scaler, [train_loader, eval_loader], logger, [writer, writer_eval])
        else:
        train_and_evaluate(rank, epoch, hps, [net_g, net_d], [optim_g, optim_d], [scheduler_g, scheduler_d], scaler, [train_loader, None], None, None)
        scheduler_g.step()
        scheduler_d.step()

    trainer = pl.Trainer(limit_train_batches=100, max_epochs=1)
    trainer.fit(model=autoencoder, train_dataloaders=train_loader)

if __name__ == "__main__":
  main()
