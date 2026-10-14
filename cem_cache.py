"""Training cache of the CEM data set (fiss_plus_planner Collect_Data_For_ML with CEM_CPP).

The CEM data set stores every candidate of every planning cycle, one folder per scenario:
    <data>/cem_data/<scenario>/{contexts,candidates,proposals}.parquet, scenario.json
    <data>/imgs/<scenario>/<time_step>.png
This script converts it once into flat arrays that the training reads with memory maps, so a
training job (also a resumed one) starts in seconds and needs little RAM:
    cycles.parquet    one row per planning cycle: scenario outcome, search space (d/v/T min/max),
                      ego state and task (EGO_STATE: speed, acceleration, Frenet d and derivatives,
                      v_des), its feasible candidates (cand_start, cand_count), image history (frame_0..2)
    cand_z.bin        float32 (N, 3): feasible candidates z = [d, v, T]
    cand_jn.bin       float32 (N,):   J normalised within the cycle, (J - J_min) / (J_90 - J_min)
    cand_logq.bin     float32 (N,):   log density of z under CEM's sampling distribution
    images.bin        uint8 (F, H, W): grayscale BEV frames
    route_xy.bin      float32 (P, 2): the reference path ahead of every cycle (conditions.parquet, ego frame,
    route_offsets.bin int64 (cycles + 1): points every 0.1 m), cycle i = route_xy[offsets[i]:offsets[i + 1]];
                      the route condition of the CVAE (converted by cem_dataset.route_features when it is
                      read, as in the planner); only for collections with conditions.parquet (v2 on)
    meta.json         shapes, source and build settings

CEM draws a cycle's candidates from one diagonal Gaussian per iteration (in unit coordinates of
the cycle's search space). Their density is the mixture of these Gaussians (balance heuristic,
Veach & Guibas 1995); the training divides by it to undo CEM's concentration of samples near
the optimum, so the cost weight alone decides how often a candidate is learned.

python -m CVAE_efficient_sampling.cem_cache --data <OUTPUT_DIR of the collection> --out <cache dir>
python -m CVAE_efficient_sampling.cem_cache --data <...> --out <existing cache> --add_route   # only the route
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
# Ego state and task of a cycle (contexts.parquet), a possible vector condition of the CVAE:
# v, a (Cartesian), s_d, s_dd, d, d_d, d_dd (Frenet start state), v_des (desired speed of J_V)
EGO_STATE = ("v", "a", "s_d", "s_dd", "d", "d_d", "d_dd", "v_des")
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


def read_route(data_dir: str, scenario: str, steps) -> list:
    """Reference path ahead (n, 2) of each cycle in steps; None without conditions.parquet."""
    path = os.path.join(data_dir, "cem_data", scenario, "conditions.parquet")
    if not os.path.exists(path):
        return None
    ref = pd.read_parquet(path).set_index("time_step")
    return [np.column_stack([ref.at[t, "ref_x"], ref.at[t, "ref_y"]]).astype(np.float32) for t in steps]


class RouteWriter:
    """Appends the cycles' reference paths to route_xy.bin; route_offsets.bin at close()."""

    def __init__(self, out_dir: str, suffix: str = ""):
        self.out_dir, self.suffix = out_dir, suffix
        self.fh, self.offsets = open(os.path.join(out_dir, f"route_xy.bin{suffix}"), "wb"), [0]

    def add(self, paths: list) -> None:
        for p in paths:
            self.fh.write(np.ascontiguousarray(p, dtype=np.float32).tobytes())
            self.offsets.append(self.offsets[-1] + len(p))

    def close(self, keep: bool = True) -> None:
        self.fh.close()
        if not keep:
            os.remove(self.fh.name)
            return
        np.asarray(self.offsets, dtype=np.int64).tofile(os.path.join(self.out_dir, f"route_offsets.bin{self.suffix}"))


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
        ego = {f"ego_{k}": float(ctx.at[t, k]) for k in EGO_STATE}
        cycles.append({"scenario": scenario, "time_step": int(t), "cand_count": n, **bounds, **ego,
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
    return cycles, cat(z_all, (0, 3)), cat(jn_all, (0,)), cat(logq_all, (0,)), frames, read_route(data_dir, scenario, steps)


def build_cache(data_dir: str, out_dir: str, workers: int = 8, limit: int = None) -> None:
    os.makedirs(out_dir, exist_ok=True)
    done = set(f[:-5] for f in os.listdir(os.path.join(data_dir, "completed")) if f.endswith(".done"))
    scenarios = sorted(s for s in os.listdir(os.path.join(data_dir, "cem_data"))
                       if s in done and os.path.isdir(os.path.join(data_dir, "cem_data", s)))[:limit]
    files = {name: open(os.path.join(out_dir, f"{name}.bin"), "wb") for name in ("cand_z", "cand_jn", "cand_logq", "images")}
    route_writer = RouteWriter(out_dir)
    rows, skipped, n_cand, n_frames, frame_shape, t0 = [], {}, 0, 0, None, time.time()
    has_route = True
    with Pool(workers) as pool:
        for k, result in enumerate(pool.imap(process_scenario, ((data_dir, s) for s in scenarios), chunksize=4)):
            if len(result) == 2:
                skipped[result[0]] = result[1]
                print(f"skipped {result[0]}: {result[1]}", flush=True)
                continue
            cycles, z, jn, logq, frames, route = result
            has_route &= route is not None
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
            if route is not None:
                route_writer.add(route)
            if (k + 1) % 200 == 0:
                print(f"{k + 1}/{len(scenarios)} scenarios, {n_cand / 1e6:.1f} M candidates, {time.time() - t0:.0f} s", flush=True)
    for fh in files.values():
        fh.close()
    route_writer.close(keep=has_route)  # not for a collection without conditions.parquet (v1)
    pd.DataFrame(rows).to_parquet(os.path.join(out_dir, "cycles.parquet"), index=False)
    info_path = os.path.join(data_dir, "cem_data", "dataset_info.json")
    meta = {"source": os.path.abspath(data_dir), "scenarios": len(scenarios) - len(skipped), "skipped": skipped,
            "cycles": len(rows),
            "candidates": n_cand, "frames": n_frames, "frame_shape": list(frame_shape), "history": HISTORY,
            "j_scale_quantile": J_SCALE_QUANTILE, "z_order": ["d", "v", "T"],
            "route": {"points": route_writer.offsets[-1], "source": "conditions.parquet"} if has_route else None,
            "dataset_info": json.load(open(info_path)) if os.path.exists(info_path) else None}
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=1)
    print(f"cache: {len(scenarios) - len(skipped)} scenarios ({len(skipped)} skipped), {len(rows)} cycles, {n_cand} feasible candidates, "
          f"{n_frames} frames, {time.time() - t0:.0f} s -> {out_dir}")


def route_of_scenario(args):
    data_dir, scenario, steps = args
    return read_route(data_dir, scenario, steps)


def add_route(data_dir: str, cache_dir: str, workers: int = 8) -> None:
    """Writes the route files of an existing cache (rows of its cycles.parquet) from the collection it was built from."""
    cycles = pd.read_parquet(os.path.join(cache_dir, "cycles.parquet"), columns=["scenario", "time_step"])
    groups = [(s, g.time_step.tolist()) for s, g in cycles.groupby("scenario", sort=False)]
    assert sum(len(t) for _, t in groups) == len(cycles) and \
        [s for s, _ in groups] == cycles.scenario.drop_duplicates().tolist(), "cycles of a scenario must be contiguous"
    writer, t0 = RouteWriter(cache_dir, suffix=".tmp"), time.time()
    with Pool(workers) as pool:
        for (scenario, _), route in zip(groups, pool.imap(route_of_scenario, ((data_dir, s, t) for s, t in groups),
                                                            chunksize=4)):
            if route is None:
                writer.close(keep=False)
                raise SystemExit(f"{scenario}: no conditions.parquet in {data_dir}")
            writer.add(route)
    writer.close()
    for name in ("route_xy.bin", "route_offsets.bin"):
        os.replace(os.path.join(cache_dir, name + ".tmp"), os.path.join(cache_dir, name))
    meta_path = os.path.join(cache_dir, "meta.json")
    meta = json.load(open(meta_path))
    meta["route"] = {"points": writer.offsets[-1], "source": "conditions.parquet"}
    json.dump(meta, open(meta_path, "w"), indent=1)
    print(f"route: {len(cycles)} cycles of {len(groups)} scenarios, {writer.offsets[-1]} points, "
          f"{time.time() - t0:.0f} s -> {cache_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="OUTPUT_DIR of the CEM collection (contains cem_data/, imgs/)")
    parser.add_argument("--out", required=True, help="cache directory")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, help="only the first n scenarios (quick tests)")
    parser.add_argument("--add_route", action="store_true", help="only add the route files to the existing cache --out")
    args = parser.parse_args()
    if args.add_route:
        add_route(args.data, args.out, args.workers)
    else:
        build_cache(args.data, args.out, args.workers, args.limit)
