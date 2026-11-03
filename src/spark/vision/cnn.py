"""SparkNet: a family of small CNNs for one question, "is this bay occupied?".

The family is parameterised by a width multiplier and the input resolution, so the
benchmark can trace accuracy against compute instead of guessing a size. Blocks are
depthwise-separable convolutions with ReLU6, which quantise cleanly to int8.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


def _ch(c: float, divisor: int = 4) -> int:
    return max(divisor, int(c + divisor / 2) // divisor * divisor)


class DWSeparable(nn.Sequential):
    def __init__(self, cin: int, cout: int, stride: int) -> None:
        super().__init__(
            nn.Conv2d(cin, cin, 3, stride, 1, groups=cin, bias=False),
            nn.BatchNorm2d(cin),
            nn.ReLU6(inplace=True),
            nn.Conv2d(cin, cout, 1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU6(inplace=True),
        )


@dataclass(frozen=True)
class SparkNetConfig:
    width: float = 1.0
    input_size: int = 64

    @property
    def name(self) -> str:
        return f"sparknet-w{self.width:g}-r{self.input_size}"


# (output channels at width 1.0, stride)
_BLOCKS: tuple[tuple[int, int], ...] = ((32, 1), (64, 2), (64, 1), (128, 2), (128, 1), (256, 2))


class SparkNet(nn.Module):
    def __init__(self, config: SparkNetConfig, dropout: float = 0.2) -> None:
        super().__init__()
        self.config = config
        c = _ch(16 * config.width)
        layers: list[nn.Module] = [
            nn.Conv2d(3, c, 3, 2, 1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU6(inplace=True),
        ]
        for out, stride in _BLOCKS:
            o = _ch(out * config.width)
            layers.append(DWSeparable(c, o, stride))
            c = o
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(c, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool(self.features(x)))  # (n, 1) logits


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def count_macs(model: nn.Module, input_size: int) -> int:
    """Multiply-accumulates for one patch, counted with forward hooks on conv and linear."""
    total = 0

    def conv_hook(m: nn.Conv2d, _inp: tuple[torch.Tensor, ...], out: torch.Tensor) -> None:
        nonlocal total
        k = m.kernel_size[0] * m.kernel_size[1] * (m.in_channels // m.groups)
        total += int(out.numel() // out.shape[0]) * k

    def lin_hook(m: nn.Linear, _inp: tuple[torch.Tensor, ...], _out: torch.Tensor) -> None:
        nonlocal total
        total += m.in_features * m.out_features

    hooks = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(lin_hook))
    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 3, input_size, input_size))
    model.train(was_training)
    for h in hooks:
        h.remove()
    return total
