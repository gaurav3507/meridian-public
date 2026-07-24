"""Simulator config: measurement_nuisance.

Site variation lives in the MEASUREMENT process (per-region HRF latency /
dispersion / amplitude that differ by site), NOT in the causal DAG. All sites
share the SAME base mechanism graph. This is the regime the two-compartment
split targets: z_meas should absorb the per-region site signature while z_mech
stays site-invariant.

Small diagnostic size: 10 sites x 30 subjects x 1 seed. Saved in the finalized
format (loader-compatible) as sim_measurement_nuisance_d10_seed0.npz.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from generate_dag import generate_dag
from dynamics import simulate_latents
from hrf import make_mixing_matrix, double_gamma_kernel, TR
from noise import add_noise

ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "data" / "simulated"

D, N, T = 10, 50, 200
N_SITES, SUBJ_PER_SITE, SEED = 10, 30, 0
MASTER_SEED = 0
HRF_TIME_LENGTH = 32.0
SNR_TARGET, THERMAL_FRAC, N_DRIFT = 1.0, 0.2, 3

# site HRF nuisance magnitudes (per-region, site-consistent)
DELAY_STD = 1.5        # seconds of peak-latency jitter
DISP_STD = 0.25
AMP_STD = 0.30


def site_hrf_params(site: int):
    rng = np.random.default_rng([MASTER_SEED, 777, site])
    delays = np.clip(6.0 + rng.normal(0, DELAY_STD, N), 3.0, 9.0)
    disps = np.clip(1.0 + rng.normal(0, DISP_STD, N), 0.5, 2.0)
    amps = np.clip(1.0 + rng.normal(0, AMP_STD, N), 0.3, None)
    return delays, disps, amps


def build_site_kernels(delays, disps):
    return [double_gamma_kernel(TR, HRF_TIME_LENGTH, peak_delay=float(delays[r]),
                                dispersion=float(disps[r])) for r in range(N)]


def forward_regionwise(latents, W, kernels, amps):
    mixed = latents @ W                                  # (T, N)
    bold = np.empty((T, N), dtype=np.float64)
    for r in range(N):
        sig = mixed[:, r] * amps[r]
        bold[:, r] = np.convolve(sig, kernels[r], "full")[:T]
    return bold


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    base_dag = generate_dag(d=D, edge_density=1.5, max_lag=2, seed=MASTER_SEED)
    W = make_mixing_matrix(D, N, seed=MASTER_SEED + 1)

    # per-site measurement parameters + kernels (fixed across that site's subjects)
    site_delays = np.zeros((N_SITES, N)); site_disps = np.zeros((N_SITES, N))
    site_amps = np.zeros((N_SITES, N)); site_kernels = []
    for s in range(N_SITES):
        d_, p_, a_ = site_hrf_params(s)
        site_delays[s], site_disps[s], site_amps[s] = d_, p_, a_
        site_kernels.append(build_site_kernels(d_, p_))

    n_total = N_SITES * SUBJ_PER_SITE
    ts = np.empty((n_total, T, N), dtype=np.float32)
    Z = np.empty((n_total, T, D), dtype=np.float32)
    site_ids = np.empty(n_total, dtype="U8")
    subject_ids = np.empty(n_total, dtype="U48")

    root_ss = np.random.SeedSequence([MASTER_SEED,
                                      hash("measurement_nuisance") & 0xFFFF, SEED])
    idx = 0
    for s in range(N_SITES):
        for k in range(SUBJ_PER_SITE):
            subj_ss, noise_ss = root_ss.spawn(1)[0].spawn(2)
            lat = simulate_latents(base_dag, T=T, noise_scale=1.0,
                                   burn_in=50, seed=subj_ss)
            clean = forward_regionwise(lat, W, site_kernels[s], site_amps[s])
            noisy, _ = add_noise(clean, snr_target=SNR_TARGET,
                                 thermal_frac=THERMAL_FRAC,
                                 n_drift_components=N_DRIFT, seed=noise_ss)
            ts[idx] = noisy.astype(np.float32)
            Z[idx] = lat.astype(np.float32)
            site_ids[idx] = f"SITE{s:02d}"
            subject_ids[idx] = f"measurement_nuisance_s{SEED}_SITE{s:02d}_{k:03d}"
            idx += 1

    # all sites share the SAME mechanism graph (no cross-site DAG shift)
    ground_truth_dags = np.stack([base_dag.A for _ in range(N_SITES)])

    out = OUTDIR / f"sim_measurement_nuisance_d{D}_seed{SEED}.npz"
    np.savez_compressed(
        out,
        ts=ts, site_ids=site_ids, subject_ids=subject_ids, latents=Z,
        config_name="measurement_nuisance",
        cross_site=np.bool_(False), temporal=np.bool_(False),
        n_edges_shifted=np.int64(0), sparsity_level="none",
        temporal_strength=np.float64(0.0), seed=np.int64(SEED),
        ground_truth_dags=ground_truth_dags,
        base_A=base_dag.A, base_topo_order=base_dag.topo_order, mixing_W=W,
        d=np.int32(D), n_regions=np.int32(N), T=np.int32(T),
        max_lag=np.int32(base_dag.max_lag), TR=np.float64(TR),
        hrf_time_length=np.float64(HRF_TIME_LENGTH),
        region_specific_hrf=np.bool_(True),
        n_drift_components=np.int64(N_DRIFT),
        snr_target=np.float64(SNR_TARGET), thermal_frac=np.float64(THERMAL_FRAC),
        # measurement ground truth (for later z_meas analysis)
        site_hrf_delays=site_delays.astype(np.float32),
        site_hrf_dispersions=site_disps.astype(np.float32),
        site_hrf_amplitudes=site_amps.astype(np.float32),
    )
    mb = out.stat().st_size / 1e6
    print(f"wrote {out.name}  ts={ts.shape}  {mb:.1f} MB")
    print(f"  {N_SITES} sites x {SUBJ_PER_SITE} subjects; shared base DAG "
          f"({base_dag.n_edges()} edges); per-site per-region HRF nuisance")
    print(f"  site HRF peak-delay range: "
          f"[{site_delays.min():.1f}, {site_delays.max():.1f}] s")

    # quick check: is site linearly decodable from a simple summary of ts?
    # (per-region lag-1 AC — a measurement-shape feature that survives z-scoring)
    def lag1(x):
        xc = x - x.mean(1, keepdims=True)
        num = (xc[:, :-1] * xc[:, 1:]).sum(1); den = (xc * xc).sum(1)
        return num / np.where(den > 0, den, 1)
    feat = lag1(ts)                                     # (n, N)
    from numpy.linalg import lstsq
    sids = np.array([int(s[-2:]) for s in site_ids])
    # crude separability: variance of per-site mean lag-1 AC across sites
    site_means = np.stack([feat[sids == s].mean(0) for s in range(N_SITES)])
    between = site_means.var(0).mean(); within = np.stack(
        [feat[sids == s].var(0) for s in range(N_SITES)]).mean()
    print(f"  lag-1 AC site separability (between/within var) = "
          f"{between / within:.3f}  (>0 => site signature present post-shape)")


if __name__ == "__main__":
    main()
