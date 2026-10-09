"""Quality of generated samples on planning cycles of the CEM cache (training per epoch and
test_cost_cvae.py).

For every cycle, K samples of a sampler are compared with the cycle's feasible CEM candidates and
with the cycle's target distribution (the one the CVAE is trained on, cem_dataset.py):
    in_bounds    share of samples inside the cycle's search space
    best_dist    distance of the closest sample to the cycle's best CEM candidate (in std units)
    best_cost    J~ of the feasible candidate nearest to the best sample (0 = CEM's best, 1 = J_90),
                 a proxy of the cost the planner would reach with these K samples ("best cost@K")
    mean_dist    distance of the samples' mean to the target mean (std units): does it use the scene?
    spread       std of the samples (std units, mean over d, v, T); much smaller than target_spread
                 means the CVAE collapsed to one point per scene (posterior collapse)
"""
import numpy as np

METRICS = ("in_bounds", "best_dist", "best_cost", "mean_dist", "spread")


def generation_metrics(samplers: dict, cache, rows, k: int, tau: float, density_correction: bool,
                       std: np.ndarray, seed: int = 0) -> dict:
    """samplers: name -> fn(frames, lo, hi, k, rng) returning (k, 3) samples [d, v, T].
    Returns {name: {metric: value}} (medians; means for in_bounds and spread) and target_spread."""
    rng = np.random.default_rng(seed)
    res = {name: {m: [] for m in METRICS} for name in samplers}
    target_spread = []
    for row in rows:
        r = cache.cycles.iloc[row]
        lo, hi = np.array([r.d_min, r.v_min, r.T_min]), np.array([r.d_max, r.v_max, r.T_max])
        s, n = r.cand_start, r.cand_count
        cand = np.asarray(cache.z[s:s + n], dtype=np.float64) / std
        jn = np.asarray(cache.jn[s:s + n], dtype=np.float64)
        logw = -jn / tau - (np.asarray(cache.logq[s:s + n], dtype=np.float64) if density_correction else 0.0)
        w = np.exp(logw - logw.max())
        target_mean = (w[:, None] * cand).sum(0) / w.sum()
        target_spread.append(np.sqrt((w[:, None] * (cand - target_mean) ** 2).sum(0) / w.sum()).mean())
        best = cand[np.argmin(jn)]
        frames = [cache.images[r[f"frame_{h}"]] for h in range(3)]
        for name, sample in samplers.items():
            z = sample(frames, lo, hi, k, rng)
            zs = z / std
            nearest = np.argmin(((zs[:, None, :] - cand[None]) ** 2).sum(-1), axis=1)
            out = res[name]
            out["in_bounds"].append(np.mean(np.all((z >= lo - 1e-6) & (z <= hi + 1e-6), axis=1)))
            out["best_dist"].append(np.min(np.linalg.norm(zs - best, axis=1)))
            out["best_cost"].append(np.min(jn[nearest]))
            out["mean_dist"].append(np.linalg.norm(zs.mean(0) - target_mean))
            out["spread"].append(zs.std(0).mean())
    summary = {name: {m: float(np.mean(v) if m in ("in_bounds", "spread") else np.median(v)) for m, v in out.items()}
               for name, out in res.items()}
    summary["target_spread"] = float(np.mean(target_spread))
    return summary
