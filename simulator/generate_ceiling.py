"""Identifiability-ceiling dataset: the easiest possible identifying regime.

  * dense cross-site mechanism shift (8 of 15 edges) with LARGE magnitude
  * 20 environments (sites)
  * NO HRF (BOLD = linear mixing of latents)
  * NO observation noise
  * shared base DAG known; per-site DAGs known

If MERIDIAN's encoder/decoder + group-sparse SMS cannot recover the latents
here, the failure is architectural rather than signal-strength.
Saved loader-compatible as sim_ceiling_dense_d10_seed0.npz.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from generate_dag import generate_dag
from hrf import make_mixing_matrix, TR
from variation import make_site_dag, generate_subject

ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "data" / "simulated"

D, N, T = 10, 50, 200
N_SITES, SUBJ_PER_SITE, SEED, MASTER = 20, 40, 0, 0
N_EDGES_SHIFT = 8            # dense (of 15 base edges)
SITE_SHIFT_SCALE = 1.5      # large per-edge shift magnitude


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    base = generate_dag(D, 1.5, 2, seed=MASTER)
    W = make_mixing_matrix(D, N, seed=MASTER + 1)
    gt_dags = np.stack([make_site_dag(base, s, N_EDGES_SHIFT, SITE_SHIFT_SCALE,
                                      base_seed=MASTER).A for s in range(N_SITES)])

    n_total = N_SITES * SUBJ_PER_SITE
    ts = np.empty((n_total, T, N), dtype=np.float32)
    Z = np.empty((n_total, T, D), dtype=np.float32)
    site_ids = np.empty(n_total, dtype="U8")
    subject_ids = np.empty(n_total, dtype="U48")

    root = np.random.SeedSequence([MASTER, hash("ceiling") & 0xFFFF, SEED])
    idx = 0
    for s in range(N_SITES):
        for k in range(SUBJ_PER_SITE):
            subj_ss = root.spawn(1)[0]
            sim = generate_subject(
                base, s, T, cross_site=True, temporal=False,
                n_edges_shifted=N_EDGES_SHIFT, temporal_strength=0.0,
                site_shift_scale=SITE_SHIFT_SCALE, noise_scale=1.0,
                burn_in=50, base_seed=MASTER, seed=subj_ss)
            x = sim.x                              # (T, D) true latents
            ts[idx] = (x @ W).astype(np.float32)   # linear mixing, no HRF, no noise
            Z[idx] = x.astype(np.float32)
            site_ids[idx] = f"SITE{s:02d}"
            subject_ids[idx] = f"ceiling_dense_s{SEED}_SITE{s:02d}_{k:03d}"
            idx += 1

    out = OUTDIR / f"sim_ceiling_dense_d{D}_seed{SEED}.npz"
    np.savez_compressed(
        out, ts=ts, site_ids=site_ids, subject_ids=subject_ids, latents=Z,
        config_name="ceiling_dense", cross_site=np.bool_(True),
        temporal=np.bool_(False), n_edges_shifted=np.int64(N_EDGES_SHIFT),
        sparsity_level="dense", temporal_strength=np.float64(0.0),
        seed=np.int64(SEED), ground_truth_dags=gt_dags,
        base_A=base.A, base_topo_order=base.topo_order, mixing_W=W,
        d=np.int32(D), n_regions=np.int32(N), T=np.int32(T),
        max_lag=np.int32(base.max_lag), TR=np.float64(TR),
        hrf_time_length=np.float64(0.0), region_specific_hrf=np.bool_(False),
        n_drift_components=np.int64(0), snr_target=np.float64(1e9),
        thermal_frac=np.float64(0.0),
    )
    print(f"wrote {out.name}  ts={ts.shape}  {out.stat().st_size/1e6:.1f} MB")
    print(f"  {N_SITES} sites x {SUBJ_PER_SITE} subj; dense shift "
          f"{N_EDGES_SHIFT}/{base.n_edges()} edges, scale {SITE_SHIFT_SCALE}; "
          f"linear mixing, no HRF, no noise")
    spread = float(np.abs(gt_dags - gt_dags[0]).max())
    print(f"  cross-site DAG spread (max|site-site0|) = {spread:.3f} (large)")


if __name__ == "__main__":
    main()
