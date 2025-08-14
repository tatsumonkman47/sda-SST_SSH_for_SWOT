r"""Helpers"""

import h5py
import json
import math
import ot
import random
import torch
import wandb

from pathlib import Path
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from tqdm import trange
from typing import *

from .score import *


ACTIVATIONS = {
    'ReLU': torch.nn.ReLU,
    'ELU': torch.nn.ELU,
    'GELU': torch.nn.GELU,
    'SELU': torch.nn.SELU,
    'SiLU': torch.nn.SiLU,
}


def random_config(configs: Dict[str, Sequence[Any]]) -> Dict[str, Any]:
    return {
        key: random.choice(values)
        for key, values in configs.items()
    }


def save_config(config: Dict[str, Any], path: Path) -> None:
    with open(path / 'config.json', mode='x') as f:
        json.dump(config, f)


def load_config(path: Path) -> Dict[str, Any]:
    with open(path / 'config.json', mode='r') as f:
        return json.load(f)


def to(x: Any, **kwargs) -> Any:
    if torch.is_tensor(x):
        return x.to(**kwargs)
    elif type(x) is list:
        return [to(y, **kwargs) for y in x]
    elif type(x) is tuple:
        return tuple(to(y, **kwargs) for y in x)
    elif type(x) is dict:
        return {k: to(v, **kwargs) for k, v in x.items()}
    else:
        return x


class TrajectoryDataset(Dataset):
    def __init__(
        self,
        file: Path,
        window: int = None,
        flatten: bool = False,
    ):
        super().__init__()

        with h5py.File(file, mode='r') as f:
            self.data = f['x'][:]

        self.window = window
        self.flatten = flatten

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i: int) -> Tuple[Tensor, Dict]:
        x = torch.from_numpy(self.data[i])

        if self.window is not None:
            i = torch.randint(0, len(x) - self.window + 1, size=())
            x = torch.narrow(x, dim=0, start=i, length=self.window)

        if self.flatten:
            return x.flatten(0, 1), {}
        else:
            return x, {}


def generate_context(x: torch.Tensor, context_channels: int = 4) -> torch.Tensor:
    """Create 2D positional context of shape (B, context_channels, H, W) matching x"""
    B, T, C, H, W = x.shape
    y = torch.linspace(0, 1, H, device=x.device)
    x_ = torch.linspace(0, 1, W, device=x.device)
    grid_y, grid_x = torch.meshgrid(y, x_, indexing="ij")

    # Sinusoidal positional encodings
    pos = torch.stack([
        torch.cos(2 * torch.pi * grid_x),
        torch.sin(2 * torch.pi * grid_x),
        torch.cos(2 * torch.pi * grid_y),
        torch.sin(2 * torch.pi * grid_y),
    ])  # shape (4, H, W)

    # Broadcast to batch
    pos = pos.unsqueeze(0).expand(B, -1, -1, -1)  # (B, 4, H, W)
    return pos

def grad_global_norm(model):
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.detach().pow(2).sum().item()
    return total ** 0.5    


def loop(
    sde: VPSDE,
    trainset: Dataset,
    validset: Dataset,
    epochs: int = 256,
    batch_size: int = 64,
    optimizer: str = 'AdamW',
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-3,
    scheduler: str = 'edm',        # NEW: default to edm-style
    device: str = 'cpu',
    num_workers: int = 20,
    log_every_n: int = 1,
    # EDM-style LR knobs (can live in cfg.training)
    lr_rampup_kimg: float = 100.0,  # warmup over 10k images
    lr_decay: float = 0.999,       # exponential step-decay base
    images_per_item: int = 1,      # set to L if you want to count frames as images
    **absorb,
) -> Iterator:

    # Data
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                             num_workers=num_workers, persistent_workers=True)
    validloader = DataLoader(validset, batch_size=batch_size, shuffle=True,
                             num_workers=num_workers, persistent_workers=True)

    # Optimizer
    if optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(sde.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError()

    # ---- LR scheduling setup ----
    use_edm = (scheduler == 'edm')
    if not use_edm:
        # keep your legacy options if you want them
        if scheduler == 'linear':
            lr_lambda = lambda t: 1 - (t / epochs)
        elif scheduler == 'cosine':
            lr_lambda = lambda t: (1 + math.cos(math.pi * t / epochs)) / 2
        elif scheduler == 'exponential':
            lr_lambda = lambda t: math.exp(-7 * (t / epochs) ** 2)
        else:
            raise ValueError(f"Unknown scheduler: {scheduler}")
        torch_sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # For EDM schedule
    lr_base = learning_rate
    cur_nimg = 0.0                      # total images seen so far
    nimg_ramp = max(lr_rampup_kimg * 1000.0, 1e-8)  # convert kimg -> images

    # Optional: define metric names up front in train()
    # wandb.define_metric("lr")
    global_step = 0

    for epoch in (bar := trange(epochs, ncols=88)):
        losses_train, losses_valid = [], []

        # ---- Train ----
        sde.train()
        for b_idx, batch in enumerate(trainloader):
            x, _ = to(batch, device=device)
            kwargs = {"c": generate_context(x)}

            # EDM-style LR update *before* forward/backward is fine (or after; just be consistent)
            if use_edm:
                # images processed this step:
                # if each dataset item has multiple frames and you want to count each frame as an image,
                # set images_per_item=L when calling loop(...)
                nimg_this_step = x.size(0) * images_per_item
                cur_nimg += nimg_this_step
                # Warmup
                ramp = min(cur_nimg / nimg_ramp, 1.0)
                new_lr = lr_base * ramp
                # Step-decay every 5e6 images (matches your snippet)
                decay_steps = (cur_nimg - nimg_ramp) // 5e6 if cur_nimg > nimg_ramp else 0
                if decay_steps > 0:
                    new_lr *= (lr_decay ** decay_steps)
                for g in optimizer.param_groups: # type: ignore
                    g["lr"] = float(new_lr)
                # Log LR by images seen (nice x-axis for W&B)
                wandb.log({"lr": new_lr, "images_seen": cur_nimg, "step": global_step}, step=global_step)

            # LOGGING (sanity stats once)
            if b_idx == 0 and epoch == 0:
                wandb.log({
                    "debug/x/mean_per_channel": x.detach().mean(dim=(0,2,3)).cpu().numpy(),
                    "debug/x/std_per_channel":  x.detach().std (dim=(0,2,3)).cpu().numpy(),
                    "debug/c/mean":             kwargs["c"].detach().mean().item(),
                    "debug/c/std":              kwargs["c"].detach().std().item(),
                }, commit=False)

            l = sde.loss(x, **kwargs)
            l.backward()

            # Grad norm logging
            gn = grad_global_norm(sde)
            wandb.log({"grad_norm": gn, "epoch": epoch, "batch_idx": b_idx, "step": global_step}, step=global_step)
            optimizer.step() # type: ignore
            optimizer.zero_grad() # type: ignore
            losses_train.append(l.detach())
            # Batch-wise logging (train)
            if (b_idx % log_every_n) == 0:
                wandb.log(
                    {
                        "batch_loss/train": l.item(),
                        "lr": optimizer.param_groups[0]["lr"], # type: ignore
                        "epoch": epoch,
                        "batch_idx": b_idx,
                        "step": global_step,
                    },
                    step=global_step,
                )
            global_step += 1

        # ---- Valid ----
        sde.eval()
        with torch.no_grad():
            for vb_idx, batch in enumerate(validloader):
                x, _ = to(batch, device=device)
                kwargs = {"c": generate_context(x)}
                lv = sde.loss(x, **kwargs)
                losses_valid.append(lv)

                if (vb_idx % log_every_n) == 0:
                    wandb.log(
                        {
                            "batch_loss/valid": lv.item(),
                            "epoch": epoch,
                            "batch_idx": vb_idx,
                            "step": global_step,
                        },
                        step=global_step,
                    )

        # ---- Epoch stats ----
        loss_train = torch.stack(losses_train).mean().item()
        loss_valid = torch.stack(losses_valid).mean().item()
        lr_now = optimizer.param_groups[0]['lr']

        wandb.log(
            {
                "loss_train/epoch": loss_train,
                "loss_valid/epoch": loss_valid,
                "lr/epoch": lr_now,
                "epoch": epoch,
                "step": global_step,
            },
            step=global_step,
        )
        bar.set_description(f"Epoch {epoch+1}/{epochs} | Train Loss: {loss_train:.4f} | Valid Loss: {loss_valid:.4f} | LR: {lr_now:.6f}")
        # ---- Checkpointing ----
        if (epoch + 1) % 10 == 0:  # Save every 10 epochs (adjust as needed)
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": sde.score.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(), # type: ignore
                "global_step": global_step,
                "loss_train": loss_train,
                "loss_valid": loss_valid,
                "lr": lr_now,
            }
            torch.save(checkpoint, f"checkpoint_epoch_{epoch+1}.pth")

        yield loss_train, loss_valid, lr_now

        bar.set_postfix(lt=loss_train, lv=loss_valid, lr=lr_now)

        # Legacy schedulers (not used for 'edm')
        if not use_edm:
            torch_sched.step()
            


def bpf(
    x: Tensor,  # (M, *)
    y: Tensor,  # (N, *)
    transition: Callable[[Tensor], Tensor],
    likelihood: Callable[[Tensor, Tensor], Tensor],
    step: int = 1,
) -> Tensor:  # (M, N + 1, *)
    r"""Performs bootstrap particle filter (BPF) sampling

    .. math:: p(x_0, x_1, ..., x_n | y_1, ..., y_n)
        = p(x_0) \prod_i p(x_i | x_{i-1}) p(y_i | x_i)

    Wikipedia:
        https://wikipedia.org/wiki/Particle_filter

    Arguments:
        x: A set of initial states :math:`x_0`.
        y: The vector of observations :math:`(y_1, ..., y_n)`.
        transition: The transition function :math:`p(x_i | x_{i-1})`.
        likelihood: The likelihood function :math:`p(y_i | x_i)`.
        step: The number of transitions per observation.
    """

    x = x[:, None]

    for yi in y:
        for _ in range(step):
            xi = transition(x[:, -1])
            x = torch.cat((x, xi[:, None]), dim=1)

        w = likelihood(yi, xi)
        j = torch.multinomial(w, len(w), replacement=True)
        x = x[j]

    return x


def emd(
    x: Tensor,  # (M, *)
    y: Tensor,  # (N, *)
) -> Tensor:
    r"""Computes the earth mover's distance (EMD) between two distributions.

    Wikipedia:
        https://wikipedia.org/wiki/Earth_mover%27s_distance

    Arguments:
        x: A set of samples :math:`x ~ p(x)`.
        y: A set of samples :math:`y ~ q(y)`.
    """

    return ot.emd2(
        x.new_tensor(()),
        y.new_tensor(()),
        torch.cdist(x.flatten(1), y.flatten(1)),
    )


def mmd(
    x: Tensor,  # (M, *)
    y: Tensor,  # (N, *)
) -> Tensor:
    r"""Computes the empirical maximum mean discrepancy (MMD) between two distributions.

    Wikipedia:
        https://wikipedia.org/wiki/Kernel_embedding_of_distributions

    Arguments:
        x: A set of samples :math:`x ~ p(x)`.
        y: A set of samples :math:`y ~ q(y)`.
    """

    x = x.flatten(1)
    y = y.flatten(1)

    xx = x @ x.T
    yy = y @ y.T
    xy = x @ y.T

    dxx = xx.diag().unsqueeze(1)
    dyy = yy.diag().unsqueeze(0)

    err_xx = dxx + dxx.T - 2 * xx
    err_yy = dyy + dyy.T - 2 * yy
    err_xy = dxx + dyy - 2 * xy

    mmd = 0

    for sigma in (1e-3, 1e-2, 1e-1, 1e-0, 1e1, 1e2, 1e3):
        kxx = torch.exp(-err_xx / sigma)
        kyy = torch.exp(-err_yy / sigma)
        kxy = torch.exp(-err_xy / sigma)

        mmd = mmd + kxx.mean() + kyy.mean() - 2 * kxy.mean()

    return mmd
