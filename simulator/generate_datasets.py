"""Part 6 - Dataset generation for the MERIDIAN fMRI simulator.

Wires Parts 1-5 into five labelled datasets. All configs share ONE ground-truth
base DAG and ONE mixing matrix (so cross-site vs temporal comparisons are on the
same underlying causal structure / observation map). Per config we vary which
variation knobs are active:

  sparse_crosssite : cross-site ON (2 edges shifted),      temporal OFF
  temporal_only    : cross-site OFF,                        temporal ON
  crosssite_only   : cross-site ON (2 edges shifted),      temporal OFF   (== sparse; clean pair vs temporal_only)
  both_on          : cross-site ON (2 edges shifted),      temporal ON
  dense_crosssite  : cross-site ON (~half of edges),       temporal OFF

Each (config, seed) is one .npz file with 10 sites x 100 subjects. Saved in the
ABIDE-compatible layout (Part 8): key ``X`` is (subjects, T, N), plus site_ids,
config flags, and the ground-truth DAG / per-site adjacencies / mixing matrix.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from generate_dag import generate_dag, DAG
from hrf import make_mixing_matrix, forward_model, TR
from noise import add_noise
from variation import make_site_dag, generate_subject

ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "data" / "simulated"


def config_specs(n_base_edges: int) -> list[dict]:
    n_half = max(1, round(0.5 * n_base_edges))
    return [
        dict(name="sparse_crosssite", cross_site=True,  temporal=False, n_edges_shifted=2,      sparsity_level="sparse"),
        dict(name="temporal_only",    cross_site=False, temporal=True,  n_edges_shifted=0,      sparsity_level="none"),
        dict(name="crosssite_only",   cross_site=True,  temporal=False, n_edges_shifted=2,      sparsity_level="sparse"),
        dict(name="both_on",          cross_site=True,  temporal=True,  n_edges_shifted=2,      sparsity_level="sparse"),
        dict(name="dense_crosssite",  cross_site=True,  temporal=False, n_edges_shifted=n_half, sparsity_level="dense"),
    ]


def generate_one_dataset(
    cfg: dict, base_dag: DAG, W: np.ndarray, seed: int, *,
    n_sites: int, subjects_per_site: int, T: int,
    temporal_strength: float, site_shift_scale: float, n_regimes: int,
    snr_target: float, thermal_frac: float, n_drift_components: int,
    hrf_time_length: float, region_specific_hrf: bool,
    burn_in: int, master_seed: int,
) -> dict:
    d = base_dag.d
    N = W.shape[1]
    n_total = n_sites * subjects_per_site

    # Per-site static ground-truth adjacencies (fixed across seeds; == base if
    # cross-site OFF). Shape (n_sites, max_lag, d, d).
    ground_truth_dags = np.stack([
        make_site_dag(base_dag, s,
                      cfg["n_edges_shifted"] if cfg["cross_site"] else 0,
                      site_shift_scale, base_seed=master_seed).A
        for s in range(n_sites)
    ])

    ts = np.empty((n_total, T, N), dtype=np.float32)
    Z = np.empty((n_total, T, d), dtype=np.float32)
    site_ids = np.empty(n_total, dtype="U8")
    subject_ids = np.empty(n_total, dtype="U48")

    # Deterministic, independent subject streams.
    root_ss = np.random.SeedSequence(
        [master_seed, hash(cfg["name"]) & 0xFFFF, seed])

    idx = 0
    for s in range(n_sites):
        for k in range(subjects_per_site):
            subj_ss, noise_ss, hrf_ss = root_ss.spawn(1)[0].spawn(3)
            sim = generate_subject(
                base_dag, s, T,
                cross_site=cfg["cross_site"], temporal=cfg["temporal"],
                n_edges_shifted=cfg["n_edges_shifted"],
                temporal_strength=temporal_strength,
                site_shift_scale=site_shift_scale, n_regimes=n_regimes,
                noise_scale=1.0, burn_in=burn_in,
                base_seed=master_seed, seed=subj_ss,
            )
            clean = forward_model(
                sim.x, W, tr=TR,
                region_specific_hrf=region_specific_hrf,
                hrf_time_length=hrf_time_length,
                hrf_jitter_seed=hrf_ss,
            )
            noisy, _ = add_noise(
                clean, snr_target=snr_target, thermal_frac=thermal_frac,
                n_drift_components=n_drift_components, seed=noise_ss)
            ts[idx] = noisy.astype(np.float32)
            Z[idx] = sim.x.astype(np.float32)
            site_ids[idx] = f"SITE{s:02d}"
            subject_ids[idx] = f"{cfg['name']}_s{seed}_SITE{s:02d}_{k:03d}"
            idx += 1

    return dict(
        # --- observed data (ABIDE-compatible; time-series key is 'ts') ---
        ts=ts, site_ids=site_ids, subject_ids=subject_ids,
        latents=Z,
        # --- config metadata ---
        config_name=cfg["name"],
        cross_site=np.bool_(cfg["cross_site"]),
        temporal=np.bool_(cfg["temporal"]),
        n_edges_shifted=np.int64(cfg["n_edges_shifted"]),
        sparsity_level=cfg["sparsity_level"],
        temporal_strength=np.float64(temporal_strength if cfg["temporal"] else 0.0),
        seed=np.int64(seed),
        # --- ground truth (per-site adjacency tensors + generative extras) ---
        ground_truth_dags=ground_truth_dags,
        base_A=base_dag.A, base_topo_order=base_dag.topo_order,
        mixing_W=W,
        # --- simulation parameters ---
        d=np.int32(d), n_regions=np.int32(N), T=np.int32(T),
        max_lag=np.int32(base_dag.max_lag), TR=np.float64(TR),
        hrf_time_length=np.float64(hrf_time_length),
        region_specific_hrf=np.bool_(region_specific_hrf),
        n_drift_components=np.int64(n_drift_components),
        snr_target=np.float64(snr_target),
        thermal_frac=np.float64(thermal_frac),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=10)
    ap.add_argument("--N", type=int, default=50)
    ap.add_argument("--T", type=int, default=200)
    ap.add_argument("--n_sites", type=int, default=10)
    ap.add_argument("--subjects_per_site", type=int, default=100)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--master_seed", type=int, default=0)
    ap.add_argument("--temporal_strength", type=float, default=0.6)
    ap.add_argument("--site_shift_scale", type=float, default=0.5)
    ap.add_argument("--n_regimes", type=int, default=2)
    # validated (Part 7) noise defaults
    ap.add_argument("--snr_target", type=float, default=1.0)
    ap.add_argument("--thermal_frac", type=float, default=0.2)
    ap.add_argument("--n_drift_components", type=int, default=3)
    ap.add_argument("--hrf_time_length", type=float, default=32.0)
    ap.add_argument("--region_specific_hrf", action="store_true")
    ap.add_argument("--burn_in", type=int, default=50)
    ap.add_argument("--outdir", type=str, default=str(OUTDIR))
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    base_dag = generate_dag(d=args.d, edge_density=1.5, max_lag=2,
                            seed=args.master_seed)
    W = make_mixing_matrix(args.d, args.N, seed=args.master_seed + 1)
    specs = config_specs(base_dag.n_edges())

    print("=" * 72)
    print("PART 6  -  dataset generation")
    print("=" * 72)
    print(f"d={args.d}  N={args.N}  T={args.T}  sites={args.n_sites}  "
          f"subj/site={args.subjects_per_site}  seeds={args.seeds}")
    print(f"base DAG: {base_dag.n_edges()} edges; dense shift = "
          f"{specs[-1]['n_edges_shifted']} edges")
    print(f"writing to {outdir}")

    manifest = []
    t0 = time.time()
    total_jobs = len(specs) * len(args.seeds)
    with tqdm(total=total_jobs, desc="datasets", unit="file") as bar:
        for cfg in specs:
            for seed in args.seeds:
                data = generate_one_dataset(
                    cfg, base_dag, W, seed,
                    n_sites=args.n_sites,
                    subjects_per_site=args.subjects_per_site, T=args.T,
                    temporal_strength=args.temporal_strength,
                    site_shift_scale=args.site_shift_scale,
                    n_regimes=args.n_regimes, snr_target=args.snr_target,
                    thermal_frac=args.thermal_frac,
                    n_drift_components=args.n_drift_components,
                    hrf_time_length=args.hrf_time_length,
                    region_specific_hrf=args.region_specific_hrf,
                    burn_in=args.burn_in, master_seed=args.master_seed,
                )
                fname = f"sim_{cfg['name']}_d{args.d}_seed{seed}.npz"
                fpath = outdir / fname
                np.savez_compressed(fpath, **data)
                manifest.append((fname, data["ts"].shape,
                                 fpath.stat().st_size / 1e6))
                bar.update(1)

    dt = time.time() - t0
    print(f"\nGenerated {len(manifest)} files in {dt:.1f}s")
    print(f"{'file':52s} {'X shape':>18s} {'MB':>8s}")
    for fname, shape, mb in manifest:
        print(f"  {fname:50s} {str(shape):>18s} {mb:8.1f}")
    print(f"total on disk: {sum(m for _,_,m in manifest):.1f} MB")


if __name__ == "__main__":
    main()
