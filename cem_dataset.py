"""Cost-aware training samples from the CEM cache (see cem_cache.py).

One item is one planning cycle: the condition is the cycle's 3-frame BEV history and the target
one feasible candidate z = [d, v, T] of that cycle, drawn with probability

    w_i  proportional to  exp(-J~_i / tau) / q(z_i)

J~ is the cost normalised within the cycle, q CEM's sampling density. Drawing the target this way
(importance resampling) and training with the usual unweighted ELBO is, in expectation, the
cost-weighted ELBO; the CVAE learns p(z | scene) proportional to exp(-J~ / tau) over the search space.
Draws are seeded by (seed, epoch, index), so an interrupted epoch can be resumed exactly.

Optional conditions besides the frames: the ego state (EGO_COLUMNS) and the planned route, the
reference path ahead in the ego frame (route_features, as the world-model CVAE of
CVAE_trajectory_planning: resampled every ROUTE_DS m, ROUTE_POINTS points, as deltas).
"""
import json
import os
import zlib

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, Dataset, Sampler

DT = 0.1  # planning cycle [s]
# Ego state and task of a cycle in the cache (cem_cache.py EGO_STATE, same order): the optional vector
# condition of the CVAE besides the BEV frames
EGO_COLUMNS = ("ego_v", "ego_a", "ego_s_d", "ego_s_dd", "ego_d", "ego_d_d", "ego_d_dd", "ego_v_des")
# Route condition: the reference path ahead resampled every ROUTE_DS m, ROUTE_POINTS points (0..38 m),
# as in the world-model CVAE (CVAE_trajectory_planning inference.py: ds = 2 m, 20 points, deltas)
ROUTE_POINTS = 20
ROUTE_DS = 2.0


def resample_route(path) -> np.ndarray:
    """(ROUTE_POINTS, 2) points every ROUTE_DS m of arc length along a path (n, 2) in the ego frame, from
    its first point (the reference point nearest to the ego); past its end the path goes on straight
    along its last segment (as the planner extends the reference line)."""
    p = np.asarray(path, dtype=np.float64).reshape(-1, 2)
    if len(p) == 0:
        p = np.zeros((1, 2))
    p = p[np.concatenate([[True], np.linalg.norm(np.diff(p, axis=0), axis=1) > 1e-9])]  # no repeated points
    target = np.arange(ROUTE_POINTS) * ROUTE_DS
    if len(p) == 1:
        return (p[0] + target[:, None] * np.array([1.0, 0.0])).astype(np.float32)
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
    pts = np.column_stack([np.interp(target, s, p[:, 0]), np.interp(target, s, p[:, 1])])
    beyond = target > s[-1]
    if beyond.any():
        direction = (p[-1] - p[-2]) / np.linalg.norm(p[-1] - p[-2])
        pts[beyond] = p[-1] + (target[beyond] - s[-1])[:, None] * direction
    return pts.astype(np.float32)


def route_features(path) -> np.ndarray:
    """Route input of the CVAE (ROUTE_POINTS, 2): the resampled path as deltas, the first point from the
    ego (its offset to the reference), each other one from the previous point, in units of ROUTE_DS."""
    pts = resample_route(path).astype(np.float64)
    return (np.diff(np.vstack([np.zeros((1, 2)), pts]), axis=0) / ROUTE_DS).astype(np.float32)


class EgoScaler:
    """Clips each ego feature (columns, a subset of EGO_COLUMNS) to the 0.1-99.9 % range of the training
    cycles (e.g. one scenario starts at 258 m/s), then standardises it."""

    def __init__(self, columns=EGO_COLUMNS):
        unknown = set(columns) - set(EGO_COLUMNS)
        assert not unknown, f"unknown ego columns {unknown}"
        self.columns = tuple(columns)

    def select(self, full: np.ndarray) -> np.ndarray:
        """The scaler's columns of a full ego vector (EGO_COLUMNS order)."""
        return np.asarray(full, dtype=np.float64)[[EGO_COLUMNS.index(c) for c in self.columns]]

    def fit(self, x: np.ndarray) -> "EgoScaler":
        self.lo, self.hi = np.percentile(x, 0.1, axis=0), np.percentile(x, 99.9, axis=0)
        c = np.clip(x, self.lo, self.hi)
        self.mean, self.std = c.mean(axis=0), np.maximum(c.std(axis=0), 1e-6)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((np.clip(x, self.lo, self.hi) - self.mean) / self.std).astype(np.float32)

    def save(self, path: str) -> None:
        json.dump({k: getattr(self, k).tolist() for k in ("lo", "hi", "mean", "std")} | {"columns": self.columns},
                  open(path, "w"), indent=1)

    @classmethod
    def load(cls, path: str) -> "EgoScaler":
        d = json.load(open(path))
        scaler = cls(d["columns"])
        for k in ("lo", "hi", "mean", "std"):
            setattr(scaler, k, np.array(d[k]))
        return scaler


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
        # DAgger data (planning.py: CVAE-FOP drove, the CEM expert labelled its states): the scenario outcome is
        # the learner's, so its failures do not make the labels before them bad (see select_cycles)
        self.dagger = bool((self.meta.get("dataset_info") or {}).get("dagger_learner"))
        # reference path ahead of each cycle (raw, ego frame), if the cache has it (cem_cache.py)
        self.has_route = bool(self.meta.get("route"))
        if self.has_route:
            self.route_offsets = mm("route_offsets", np.int64, (len(self.cycles) + 1,))
            self.route_xy = mm("route_xy", np.float32, (int(self.route_offsets[-1]), 2))

    def route(self, cycle: int) -> np.ndarray:
        """Reference path ahead (n, 2) of a cycle (row of cycles.parquet) in its ego frame."""
        return np.asarray(self.route_xy[self.route_offsets[cycle]:self.route_offsets[cycle + 1]])


def select_cycles(cycles: pd.DataFrame, drop_tail_s: float = 2.5, keep_failed: bool = True) -> np.ndarray:
    """Row indices of the cycles to learn from.

    Cycles without a feasible candidate (e.g. the failing cycle) carry nothing to learn. Rear-end
    failures (a recorded follower drives into the ego) keep their other cycles. In any other failed
    scenario the last drop_tail_s seconds before the failure led into the dead end and are dropped;
    keep_failed=False drops these scenarios entirely. Both apply to data where the CEM drove; in DAgger
    data (CEMCache.dagger) the learner failed and the expert's labels in these states are kept: pass
    drop_tail_s=0, keep_failed=True.
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
    """(target z normalised, image history, conditions) per item; draws_per_cycle items per cycle and
    epoch. conditions: dict of the model's further inputs, "e" = ego state normalised (when ego_scaler
    is set) and "r" = route_features (when route is set)."""

    def __init__(self, cache: CEMCache, rows: np.ndarray, tau: float = 0.5, density_correction: bool = True,
                 draws_per_cycle: int = 1, normalizer=None, img_size: int = None, seed: int = 0,
                 ego_scaler: EgoScaler = None, route: bool = False):
        self.cache, self.rows = cache, np.asarray(rows)
        self.tau, self.density_correction = tau, density_correction
        self.draws, self.normalizer, self.img_size, self.seed = draws_per_cycle, normalizer, img_size, seed
        self.start = cache.cycles.cand_start.to_numpy()
        self.count = cache.cycles.cand_count.to_numpy()
        self.frames = cache.cycles[[f"frame_{h}" for h in range(cache.meta["history"])]].to_numpy()
        self.ego_scaler = ego_scaler
        self.ego = cache.cycles[list(ego_scaler.columns)].to_numpy(np.float64) if ego_scaler is not None else None
        if route and not cache.has_route:
            raise ValueError(f"{cache.dir} has no route data: build it with cem_cache.py (--add_route)")
        self.route = route
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
        cond = {}
        if self.ego is not None:
            cond["e"] = torch.from_numpy(self.ego_scaler.transform(self.ego[cycle]))
        if self.route:
            cond["r"] = torch.from_numpy(route_features(self.cache.route(cycle)))
        return torch.from_numpy(z), self.images(cycle), cond

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


class CEMCycleConcat(ConcatDataset):
    """Items of several CEMCycleDatasets (e.g. the CEM data and DAgger rounds, Ross et al. 2011, trained on the
    aggregate), with the dataset interface the training uses."""

    def set_epoch(self, epoch: int) -> None:
        for d in self.datasets:
            d.set_epoch(epoch)

    @property
    def normalizer(self):
        return self.datasets[0].normalizer

    @normalizer.setter
    def normalizer(self, value):
        for d in self.datasets:
            d.normalizer = value

    def _shares(self, n: int) -> list:
        return [max(1, round(n * len(d) / len(self))) for d in self.datasets]

    def sample_targets(self, n: int, seed: int = 0) -> np.ndarray:
        return np.concatenate([d.sample_targets(m, seed + k) for k, (d, m) in enumerate(zip(self.datasets, self._shares(n)))])

    def effective_sample_size(self, n_cycles: int = 2000, seed: int = 0) -> np.ndarray:
        return np.concatenate([d.effective_sample_size(m, seed + k)
                               for k, (d, m) in enumerate(zip(self.datasets, self._shares(n_cycles)))])


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
