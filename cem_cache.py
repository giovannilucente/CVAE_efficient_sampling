"""Training cache of the CEM data set (fiss_plus_planner Collect_Data_For_ML with CEM_CPP).

The CEM data set stores every candidate of every planning cycle, one folder per scenario:
    <data>/cem_data/<scenario>/{contexts,candidates,proposals}.parquet, scenario.json
    <data>/imgs/<scenario>/<time_step>.png
This script converts it once into flat arrays that the training reads with memory maps, so a
training job (also a resumed one) starts in seconds and needs little RAM:
    cycles.parquet    one row per planning cycle: scenario outcome, search space (d/v/T min/max),
                      its feasible candidates (cand_start, cand_count), image history (frame_0..2)
    cand_z.bin        float32 (N, 3): feasible candidates z = [d, v, T]
    cand_jn.bin       float32 (N,):   J normalised within the cycle, (J - J_min) / (J_90 - J_min)
    cand_logq.bin     float32 (N,):   log density of z under CEM's sampling distribution
    images.bin        uint8 (F, H, W): grayscale BEV frames
    meta.json         shapes, source and build settings

CEM draws a cycle's candidates from one diagonal Gaussian per iteration (in unit coordinates of
the cycle's search space). Their density is the mixture of these Gaussians (balance heuristic,
Veach & Guibas 1995); the training divides by it to undo CEM's concentration of samples near
the optimum, so the cost weight alone decides how often a candidate is learned.

python -m CVAE_efficient_sampling.cem_cache --data <OUTPUT_DIR of the collection> --out <cache dir>
"""
import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from PIL import Image

HISTORY = 3          # frames per sample (t-2, t-1, t)
J_SCALE_QUANTILE = 90
EPS = 1e-9


def unit(z: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """z in unit coordinates of [lo, hi] (as SearchSpace::to_unit in C++)."""
    span = hi - lo
    return np.where(span > 1e-12, np.clip((z - lo) / np.where(span > 1e-12, span, 1.0), 0.0, 1.0), 0.5)


def log_mixture_density(u: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """log of (1/K) sum_k N(u; mean_k, diag std_k^2) for points u (n, 3), components (K, 3)."""
    z = (u[:, None, :] - mean[None]) / std[None]
    log_comp = -0.5 * (z ** 2).sum(-1) - np.log(std).sum(-1)[None] - 1.5 * np.log(2 * np.pi)
    m = log_comp.max(axis=1, keepdims=True)
    return (m[:, 0] + np.log(np.exp(log_comp - m).mean(axis=1))).astype(np.float32)


def process_scenario(args):
    """Feasible candidates, their weights' ingredients and the image frames of one scenario;
    (scenario, error) if it cannot be read."""
    data_dir, scenario = args
    try:
        return read_scenario(data_dir, scenario)
    except Exception as e:  # e.g. an unreadable image: skip the scenario, report it
        return scenario, f"{type(e).__name__}: {e}"


def read_scenario(data_dir: str, scenario: str):
    folder = os.path.join(data_dir, "cem_data", scenario)
    info = json.load(open(os.path.join(folder, "scenario.json")))
    ctx = pd.read_parquet(os.path.join(folder, "contexts.parquet")).set_index("time_step")
    cand = pd.read_parquet(os.path.join(folder, "candidates.parquet"),
                           columns=["time_step", "d", "v", "T", "feasible", "J_total"])
    prop = pd.read_parquet(os.path.join(folder, "proposals.parquet"))

    cycles, z_all, jn_all, logq_all = [], [], [], []
    feasible = cand[cand.feasible == 1]
    by_step = dict(tuple(feasible.groupby("time_step")))
    props = dict(tuple(prop.groupby("time_step")))
    for t in ctx.index:
        f = by_step.get(t)
        n = 0 if f is None else len(f)
        if n:
            z = f[["d", "v", "T"]].to_numpy(np.float64)
            j = f.J_total.to_numpy(np.float64)
            j_min = j.min()
            jn = (j - j_min) / (np.percentile(j, J_SCALE_QUANTILE) - j_min + EPS)
            c = ctx.loc[t]
            lo = np.array([c.d_min, c.v_min, c.T_min])
            hi = np.array([c.d_max, c.v_max, c.T_max])
            p = props[t]
            logq = log_mixture_density(unit(z, lo, hi), p[["mean_d", "mean_v", "mean_T"]].to_numpy(),
                                       np.maximum(p[["std_d", "std_v", "std_T"]].to_numpy(), 1e-6))
            z_all.append(z.astype(np.float32)); jn_all.append(jn.astype(np.float32)); logq_all.append(logq)
        bounds = ctx.loc[t, ["d_min", "d_max", "v_min", "v_max", "T_min", "T_max"]].astype(float).to_dict()
        cycles.append({"scenario": scenario, "time_step": int(t), "cand_count": n, **bounds,
                       "scenario_success": bool(info["success"]), "rear_end_failure": bool(info["rear_end_failure"]),
                       "failed_at_time_step": -1 if info["failed_at_time_step"] is None else int(info["failed_at_time_step"]),
                       "clearance_fallback": int(ctx.loc[t].clearance_fallback)})

    steps = list(ctx.index)
    frames = np.stack([np.asarray(Image.open(os.path.join(data_dir, "imgs", scenario, f"{t}.png")).convert("L"))
                       for t in steps])
    first = {t: i for i, t in enumerate(steps)}
    for row in cycles:  # history t-2, t-1, t clamped to the scenario's first frame (as CVAEDataset)
        for h in range(HISTORY):
            row[f"frame_{h}"] = first[max(row["time_step"] - (HISTORY - 1 - h), steps[0])]
    cat = lambda xs, shape: np.concatenate(xs) if xs else np.zeros(shape, np.float32)
    return cycles, cat(z_all, (0, 3)), cat(jn_all, (0,)), cat(logq_all, (0,)), frames


def build_cache(data_dir: str, out_dir: str, workers: int = 8, limit: int = None) -> None:
    os.makedirs(out_dir, exist_ok=True)
    done = set(f[:-5] for f in os.listdir(os.path.join(data_dir, "completed")) if f.endswith(".done"))
    scenarios = sorted(s for s in os.listdir(os.path.join(data_dir, "cem_data"))
                       if s in done and os.path.isdir(os.path.join(data_dir, "cem_data", s)))[:limit]
    files = {name: open(os.path.join(out_dir, f"{name}.bin"), "wb") for name in ("cand_z", "cand_jn", "cand_logq", "images")}
    rows, skipped, n_cand, n_frames, frame_shape, t0 = [], {}, 0, 0, None, time.time()
    with Pool(workers) as pool:
        for k, result in enumerate(pool.imap(process_scenario, ((data_dir, s) for s in scenarios), chunksize=4)):
            if len(result) == 2:
                skipped[result[0]] = result[1]
                print(f"skipped {result[0]}: {result[1]}", flush=True)
                continue
            cycles, z, jn, logq, frames = result
            for row in cycles:
                row["cand_start"] = n_cand
                n_cand += row["cand_count"]
                for h in range(HISTORY):
                    row[f"frame_{h}"] += n_frames
            frame_shape = frame_shape or frames.shape[1:]
            assert frames.shape[1:] == frame_shape, "all BEV images must have the same size"
            n_frames += len(frames)
            rows.extend(cycles)
            for name, arr in (("cand_z", z), ("cand_jn", jn), ("cand_logq", logq), ("images", frames)):
                files[name].write(np.ascontiguousarray(arr).tobytes())
            if (k + 1) % 200 == 0:
                print(f"{k + 1}/{len(scenarios)} scenarios, {n_cand / 1e6:.1f} M candidates, {time.time() - t0:.0f} s", flush=True)
    for fh in files.values():
        fh.close()
    pd.DataFrame(rows).to_parquet(os.path.join(out_dir, "cycles.parquet"), index=False)
    info_path = os.path.join(data_dir, "cem_data", "dataset_info.json")
    meta = {"source": os.path.abspath(data_dir), "scenarios": len(scenarios) - len(skipped), "skipped": skipped,
            "cycles": len(rows),
            "candidates": n_cand, "frames": n_frames, "frame_shape": list(frame_shape), "history": HISTORY,
            "j_scale_quantile": J_SCALE_QUANTILE, "z_order": ["d", "v", "T"],
            "dataset_info": json.load(open(info_path)) if os.path.exists(info_path) else None}
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=1)
    print(f"cache: {len(scenarios) - len(skipped)} scenarios ({len(skipped)} skipped), {len(rows)} cycles, {n_cand} feasible candidates, "
          f"{n_frames} frames, {time.time() - t0:.0f} s -> {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="OUTPUT_DIR of the CEM collection (contains cem_data/, imgs/)")
    parser.add_argument("--out", required=True, help="cache directory")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, help="only the first n scenarios (quick tests)")
    args = parser.parse_args()
    build_cache(args.data, args.out, args.workers, args.limit)
