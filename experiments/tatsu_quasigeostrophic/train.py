#!/usr/bin/env python
import wandb
import torch
import numpy as np
from typing import *

from sda.score import *
from sda.utils import *
from utils import *


from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf
import hydra
from pathlib import Path as SysPath  # avoid shadowing Hydra's `Path`
import sys
sys.path.append('/home.ufs/tm3076/swot_SUM03/SWOT_project/SWOT-inpainting-DL/src')
import claude_data_loaders

wandb.login()

def llc4320_dataloaders(cfg: DictConfig):
    DATASET_PATH = cfg.dataset.path
    patch_coords = np.load(f'{DATASET_PATH}/zarred_UVSST_x_y_coordinates_noland_nonan.npy')
    batch_size_global = cfg.training.batch_size
    workers = cfg.training.num_workers

    standards = {
        "mean_ssh": cfg.standards.mean_ssh,
        "std_ssh": cfg.standards.std_ssh,
        "mean_sst": cfg.standards.mean_sst,
        "std_sst": cfg.standards.std_sst,
        "extra_mean_tuning": cfg.standards.extra_mean_tuning,
    }

    time_range = cfg.dataset.time_range
    time_range = range(time_range[0],time_range[1],time_range[2])
    
    # Load all time slices
    full_dataset = torch.utils.data.ConcatDataset([
        claude_data_loaders.llc4320_dataset(
            DATASET_PATH, 
            t,  
            cfg.dataset.Number_timesteps, 
            patch_coords,
            cfg.dataset.infields, 
            cfg.dataset.outfields,
            cfg.dataset.in_mask_list, 
            cfg.dataset.out_mask_list,
            cfg.dataset.in_transform_list, 
            cfg.dataset.out_transform_list,
            return_masks=cfg.dataset.get("return_masks",False),
            return_metadata=cfg.dataset.get("return_metadata",False), 
            standards=standards,
            squeeze=True, L_x=512e3, L_y=512e3,
            cloud_rho=cfg.dataset.get("cloud_rho",0.2),
        ) for t in time_range
    ])
    train_len = int(0.7 * len(full_dataset))
    val_len = int(0.2 * len(full_dataset))
    test_len = len(full_dataset) - train_len - val_len
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_len, val_len, test_len])
    
    return train_dataset, val_dataset
    

    
@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def train(cfg: DictConfig):
    # Wandb stuff
    run = wandb.init(project=cfg.project_name, config=OmegaConf.to_container(cfg, resolve=True))
    wandb.define_metric("step")   # used for batch-wise metrics
    wandb.define_metric("epoch")  # used for epoch-wise metrics
    # Batch-wise metrics (you log "batch_loss/train" and "batch_loss/valid" every batch)
    wandb.define_metric("batch_loss/*", step_metric="step")
    # Epoch-wise metrics (you log these once per epoch)
    wandb.define_metric("loss_train/epoch", step_metric="epoch")
    wandb.define_metric("loss_valid/epoch", step_metric="epoch")
    wandb.define_metric("lr/epoch",         step_metric="epoch")
    
    runpath = SysPath(cfg.run_dir) / f'{run.name}_{run.id}'
    runpath.mkdir(parents=True, exist_ok=True)
    save_config(OmegaConf.to_container(cfg, resolve=True), runpath)

    # Network
    score = make_score(**cfg.model)
    sde = VPSDE(score, shape=tuple(cfg.model.input_shape)).cuda()

    # Data
    train_dataset, val_dataset = llc4320_dataloaders(cfg)

    # Training loop generator
    generator = loop(
        sde,
        train_dataset.dataset,
        val_dataset.dataset,
        device='cuda',
        **cfg.training,
    )

    for loss_train, loss_valid, lr in generator:
        run.log({
            'loss_train': loss_train,
            'loss_valid': loss_valid,
            'lr': lr,
        })

    # Save checkpoint
    torch.save(score.state_dict(), runpath / f'state.pth')

    # Sampling
    c = next(iter(train_loader))[1]["c"].cuda()
    x = sde.sample((2,), c=c, steps=cfg.sampling.steps).cpu()
    q = x[:, ::4, 0]

    run.log({'samples': wandb.Image(draw(q))})
    run.finish()

if __name__ == '__main__':
    train()
