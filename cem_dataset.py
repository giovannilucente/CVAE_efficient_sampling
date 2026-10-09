"""Cost-aware training samples from the CEM cache (see cem_cache.py).

One item is one planning cycle: the condition is the cycle's 3-frame BEV history and the target
one feasible candidate z = [d, v, T] of that cycle, drawn with probability

    w_i  proportional to  exp(-J~_i / tau) / q(z_i)

J~ is the cost normalised within the cycle, q CEM's sampling density. Drawing the target this way
(importance resampling) and training with the usual unweighted ELBO is, in expectation, the
cost-weighted ELBO; the CVAE learns p(z | scene) proportional to exp(-J~ / tau) over the search space.
Draws are seeded by (seed, epoch, index), so an interrupted epoch can be resumed exactly.
"""
import json
import os
import zlib

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler

DT = 0.1  # planning cycle [s]


def bev_tensor(frames, img_size: int = None) -> torch.Tensor:
    """Condition of the CVAE from grayscale BEV frames (oldest first), uint8 (H, W) each: one channel
    per frame, inverted and scaled to [-1, 1] (the transform of CVAEDataset), resized to img_size."""
    u8 = torch.from_numpy(np.stack([np.asarray(f, dtype=np.uint8) for f in frames]))
    img = 1.0 - u8.float() * (2.0 / 255.0)
    if img_size and img.shape[-1] != img_size:
        img = F.interpolate(img[None], size=(img_size, img_size), mode="bilinear",
                            antialias=True, align_corners=False)[0]
    return img


class CEMCache:
    """Memory-mapped arrays of a cache directory."""

    def __init__(self, cache_dir: str):
        self.dir = cache_dir
        self.meta = json.load(open(os.path.join(cache_dir, "meta.json")))
        self.cycles = pd.read_parquet(os.path.join(cache_dir, "cycles.parquet"))
        n, f, shape = self.meta["candidates"], self.meta["frames"], tuple(self.meta["frame_shape"])
        mm = lambda name, dtype, s: np.memmap(os.path.join(cache_dir, f"{name}.bin"), dtype=dtype, mode="r", shape=s)
        self.z = mm("cand_z", np.float32, (n, 3))
        self.jn = mm("cand_jn", np.float32, (n,))
        self.logq = mm("cand_logq", np.float32, (n,))
        self.images = mm("images", np.uint8, (f,) + shape)


def select_cycles(cycles: pd.DataFrame, drop_tail_s: float = 2.5, keep_failed: bool = True) -> np.ndarray:
    """Row indices of the cycles to learn from.

    Cycles without a feasible candidate (e.g. the failing cycle) carry nothing to learn. Rear-end
    failures (a recorded follower drives into the ego) keep their other cycles. In any other failed
    scenario the last drop_tail_s seconds before the failure led into the dead end and are dropped;
    keep_failed=False drops these scenarios entirely.
    """
    ok = cycles.cand_count > 0
    failed = ~cycles.scenario_success & ~cycles.rear_end_failure
    if keep_failed:
        tail = failed & (cycles.time_step > cycles.failed_at_time_step - round(drop_tail_s / DT))
        ok &= ~tail
    else:
        ok &= ~failed
    return np.flatnonzero(ok.to_numpy())


def split_by_scenario(cycles: pd.DataFrame, rows: np.ndarray, val_fraction: float, seed: int = 0):
    """Train / validation rows, validation = a fixed share of the scenarios (hash of the name)."""
    h = cycles.scenario.iloc[rows].map(lambda s: zlib.crc32(f"{seed}:{s}".encode()) / 2 ** 32).to_numpy()
    return rows[h >= val_fraction], rows[h < val_fraction]


class CEMCycleDataset(Dataset):
    """(target z normalised, image history) per item; draws_per_cycle items per cycle and epoch."""

    def __init__(self, cache: CEMCache, rows: np.ndarray, tau: float = 0.5, density_correction: bool = True,
                 draws_per_cycle: int = 1, normalizer=None, img_size: int = None, seed: int = 0):
        self.cache, self.rows = cache, np.asarray(rows)
        self.tau, self.density_correction = tau, density_correction
        self.draws, self.normalizer, self.img_size, self.seed = draws_per_cycle, normalizer, img_size, seed
        self.start = cache.cycles.cand_start.to_numpy()
        self.count = cache.cycles.cand_count.to_numpy()
        self.frames = cache.cycles[[f"frame_{h}" for h in range(cache.meta["history"])]].to_numpy()
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self):
        return len(self.rows) * self.draws

    def log_weights(self, cycle: int) -> np.ndarray:
        s, n = self.start[cycle], self.count[cycle]
        logw = -self.cache.jn[s:s + n].astype(np.float64) / self.tau
        if self.density_correction:
            logw -= self.cache.logq[s:s + n]
        return logw

    def draw(self, cycle: int, rng: np.random.Generator) -> np.ndarray:
        logw = self.log_weights(cycle)
        p = np.exp(logw - logw.max())
        i = rng.choice(len(p), p=p / p.sum())
        return np.array(self.cache.z[self.start[cycle] + i], dtype=np.float32)

    def images(self, cycle: int) -> torch.Tensor:
        return bev_tensor([self.cache.images[f] for f in self.frames[cycle]], self.img_size)

    def __getitem__(self, index: int):
        cycle = self.rows[index // self.draws]
        rng = np.random.default_rng((self.seed, self.epoch, index))
        z = self.draw(cycle, rng)
        if self.normalizer is not None:
            z = self.normalizer.transform_targets(z[None])[0]
        return torch.from_numpy(z), self.images(cycle)

    def sample_targets(self, n: int, seed: int = 0) -> np.ndarray:
        """n raw targets drawn like the training items, e.g. to fit the normalizer."""
        rng = np.random.default_rng(seed)
        return np.stack([self.draw(c, rng) for c in rng.choice(self.rows, size=n)])

    def effective_sample_size(self, n_cycles: int = 2000, seed: int = 0) -> np.ndarray:
        """Kish ESS of the weights of n_cycles random cycles (how many candidates really count)."""
        out = []
        for c in np.random.default_rng(seed).choice(self.rows, size=min(n_cycles, len(self.rows)), replace=False):
            w = np.exp(self.log_weights(c) - self.log_weights(c).max())
            out.append(w.sum() ** 2 / (w ** 2).sum())
        return np.array(out)


class ResumableSampler(Sampler):
    """Shuffled order of an epoch (seeded by seed and epoch), starting at a given position."""

    def __init__(self, n: int, seed: int = 0):
        self.n, self.seed, self.epoch, self.start = n, seed, 0, 0

    def set_epoch(self, epoch: int, start: int = 0) -> None:
        self.epoch, self.start = epoch, start

    def __iter__(self):
        order = np.random.default_rng((self.seed, self.epoch)).permutation(self.n)
        return iter(order[self.start:].tolist())

    def __len__(self):
        return self.n - self.start
