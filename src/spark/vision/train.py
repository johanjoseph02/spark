"""Training for SparkNet on cached CNRPark+EXT patches.

Augmentation targets the two failure modes that matter in a real car park: lighting
(brightness, contrast, gamma, colour cast) and partial occlusion (random erasing, which
stands in for a neighbour's car, a tree or a lamp post across the bay).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray

from spark.vision.cnn import SparkNet, SparkNetConfig

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 8
    batch_size: int = 256
    lr: float = 3e-3
    weight_decay: float = 1e-4
    seed: int = 0
    augment: bool = True


def device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def to_tensor(batch: NDArray[np.uint8], dev: torch.device) -> torch.Tensor:
    """BGR uint8 NHWC -> normalised RGB float NCHW, on ``dev``."""
    x = torch.from_numpy(np.ascontiguousarray(batch[..., ::-1])).to(dev)
    x = x.permute(0, 3, 1, 2).float().div_(255.0)
    mean = torch.tensor(MEAN, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=dev).view(1, 3, 1, 1)
    return (x - mean) / std


def augment(x: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """x: RGB in [0, 1], NCHW. Runs on the training device."""
    n, _, h, w = x.shape
    dev = x.device

    def rand(*shape: int) -> torch.Tensor:
        return torch.rand(*shape, generator=gen, device="cpu").to(dev)

    flip = rand(n) < 0.5
    x = torch.where(flip.view(n, 1, 1, 1), x.flip(3), x)
    # lighting: gamma, contrast, brightness, per-channel cast
    gamma = torch.exp((rand(n, 1, 1, 1) - 0.5) * 1.2)
    x = x.clamp(1e-4, 1).pow(gamma)
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    x = (x - mean) * (0.6 + 0.8 * rand(n, 1, 1, 1)) + mean
    x = x + (rand(n, 1, 1, 1) - 0.5) * 0.4 + (rand(n, 3, 1, 1) - 0.5) * 0.1
    # small translation by padding and cropping
    pad = max(1, h // 12)
    xp = F.pad(x, (pad, pad, pad, pad), mode="replicate")
    dx, dy = (torch.randint(0, 2 * pad + 1, (2,), generator=gen)).tolist()
    x = xp[:, :, dy:dy + h, dx:dx + w]
    # occlusion: random erasing of up to a third of the patch on half the batch
    erase = rand(n) < 0.5
    if erase.any():
        frac = 0.05 + 0.28 * rand(n)
        ar = torch.exp((rand(n) - 0.5) * 1.5)
        eh = (torch.sqrt(frac * ar) * h).clamp(2, h).long()
        ew = (torch.sqrt(frac / ar) * w).clamp(2, w).long()
        y0 = (rand(n) * (h - eh + 1).float()).long()
        x0 = (rand(n) * (w - ew + 1).float()).long()
        yy = torch.arange(h, device=dev).view(1, h, 1)
        xx = torch.arange(w, device=dev).view(1, 1, w)
        box = ((yy >= y0.view(n, 1, 1)) & (yy < (y0 + eh).view(n, 1, 1))
               & (xx >= x0.view(n, 1, 1)) & (xx < (x0 + ew).view(n, 1, 1)))
        box = box & erase.view(n, 1, 1)
        fill = rand(n, 3, 1, 1)
        x = torch.where(box.unsqueeze(1), fill.expand(n, 3, h, w), x)
    return x.clamp(0, 1)


def _normalise(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


@torch.no_grad()
def predict(model: SparkNet, images: NDArray[np.uint8], batch: int = 2048) -> NDArray[np.float32]:
    dev = next(model.parameters()).device
    model.eval()
    out = []
    for i in range(0, len(images), batch):
        out.append(torch.sigmoid(model(to_tensor(images[i:i + batch], dev))).squeeze(1).cpu())
    return torch.cat(out).numpy().astype(np.float32) if out else np.zeros(0, np.float32)


def train(
    net: SparkNetConfig,
    images: NDArray[np.uint8],
    labels: NDArray[np.int64],
    train_idx: NDArray[np.int64],
    val_idx: NDArray[np.int64],
    cfg: TrainConfig | None = None,
    log: bool = True,
) -> tuple[SparkNet, dict[str, float]]:
    cfg = cfg or TrainConfig()
    torch.manual_seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    dev = device()
    model = SparkNet(net).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = math.ceil(len(train_idx) / cfg.batch_size)
    total = cfg.epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=total,
                                                pct_start=0.15)
    y_all = torch.from_numpy(labels.astype(np.float32))
    rng = np.random.default_rng(cfg.seed)
    best_acc, best_state, t0 = -1.0, None, time.time()
    for epoch in range(cfg.epochs):
        model.train()
        order = rng.permutation(train_idx)
        running = 0.0
        for s in range(steps_per_epoch):
            idx = np.sort(order[s * cfg.batch_size:(s + 1) * cfg.batch_size])
            xb = torch.from_numpy(np.ascontiguousarray(images[idx][..., ::-1])).to(dev)
            xb = xb.permute(0, 3, 1, 2).float().div_(255.0)
            if cfg.augment:
                xb = augment(xb, gen)
            xb = _normalise(xb)
            yb = y_all[idx].to(dev)
            loss = F.binary_cross_entropy_with_logits(model(xb).squeeze(1), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            running += float(loss.detach())
        p = predict(model, images[val_idx])
        acc = float(((p >= 0.5) == labels[val_idx]).mean())
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if log:
            print(f"  {net.name} epoch {epoch + 1}/{cfg.epochs} "
                  f"loss {running / steps_per_epoch:.4f} val acc {acc:.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    return model, {"val_acc": best_acc, "train_seconds": time.time() - t0}
