"""Part 3 - The two variation injectors (THE CRITICAL PART).

Two kinds of variation, designed to be *independent* by construction:

  cross-site mechanism shift   make_site_dag(base, site, n_edges_shifted)
      A sparse subset of existing edges get perturbed strengths. The result
      is a STATIC per-site adjacency. This function never reads/produces a
      time axis, so it cannot introduce temporal drift.

  temporal regime shift        make_time_varying(dag, T, temporal_strength)
      Coupling strengths switch between regimes at changepoints within one
      scan, giving A_t of shape (T, max_lag, d, d). The per-regime deviations
      are constructed to average to EXACTLY zero over the scan, so the
      time-average of A_t equals the input adjacency exactly. This function
      never reads a site index, so it cannot introduce cross-site differences.

Independence then follows structurally:
  * cross-site ON, temporal OFF  -> A_t is constant in t  (no temporal leak)
  * temporal ON, cross-site OFF  -> every subject's time-averaged mechanism
    equals the shared base exactly -> site means identical (no cross-site leak)

The _selftest() runs the mandatory 2x2 independence check.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from generate_dag import DAG, generate_dag, is_acyclic
from dynamics import coefficient_matrices

ROOT = Path(__file__).resolve().parent


# --------------------------------------------------------------------------- #
# Cross-site mechanism shift                                                   #
# --------------------------------------------------------------------------- #
def make_site_dag(
    base_dag: DAG,
    site_index: int,
    n_edges_shifted: int,
    site_shift_scale: float = 0.5,
    base_seed: int = 0,
) -> DAG:
    """Return a site-specific DAG: a sparse subset of existing edges have
    perturbed strengths. Structure (which edges exist) is unchanged, so the
    graph stays acyclic. Deterministic given (base_seed, site_index).

    n_edges_shifted == 0 returns an exact copy of the base (no site shift).
    """
    A = base_dag.A.copy()
    if n_edges_shifted <= 0 or site_shift_scale == 0.0:
        return DAG(A=A, topo_order=base_dag.topo_order.copy(),
                   d=base_dag.d, max_lag=base_dag.max_lag)

    rng = np.random.default_rng([base_seed, site_index])
    edge_pos = np.argwhere(np.abs(A) > 0)               # (n_existing, 3): l,i,j
    n_shift = min(n_edges_shifted, len(edge_pos))
    pick = rng.choice(len(edge_pos), size=n_shift, replace=False)
    for idx in pick:
        l, i, j = edge_pos[idx]
        delta = site_shift_scale * abs(A[l, i, j]) * rng.standard_normal()
        A[l, i, j] += delta

    site_dag = DAG(A=A, topo_order=base_dag.topo_order.copy(),
                   d=base_dag.d, max_lag=base_dag.max_lag)
    assert is_acyclic(site_dag), "site shift broke acyclicity (should be impossible)"
    return site_dag


# --------------------------------------------------------------------------- #
# Temporal regime shift                                                        #
# --------------------------------------------------------------------------- #
def _regime_bounds(T: int, n_regimes: int) -> list[tuple[int, int]]:
    """Split [0, T) into n_regimes contiguous segments (changepoints)."""
    edges = np.linspace(0, T, n_regimes + 1).round().astype(int)
    return [(int(edges[r]), int(edges[r + 1])) for r in range(n_regimes)]


def make_time_varying(
    dag: DAG,
    T: int,
    temporal_strength: float,
    n_regimes: int = 2,
    seed: int | None = None,
) -> np.ndarray:
    """Return A_t of shape (T, max_lag, d, d).

    Within each regime the coupling strengths are constant; they jump at the
    changepoints. Per-regime deviations are applied only on existing edges and
    are recentred so the duration-weighted time-average is exactly zero -> the
    time-average of A_t equals dag.A exactly. temporal_strength == 0 yields a
    constant (base repeated) tensor.
    """
    base = dag.A
    A_t = np.broadcast_to(base, (T,) + base.shape).copy()
    if temporal_strength == 0.0 or n_regimes < 2:
        return A_t

    rng = np.random.default_rng(seed)
    mask = (np.abs(base) > 0)
    bounds = _regime_bounds(T, n_regimes)
    durations = np.array([hi - lo for lo, hi in bounds], dtype=float)

    # Raw per-regime deviations, only on existing edges, scaled by |edge|.
    raw = np.zeros((n_regimes,) + base.shape, dtype=np.float64)
    for r in range(n_regimes):
        dev = temporal_strength * np.abs(base) * rng.standard_normal(base.shape)
        raw[r] = np.where(mask, dev, 0.0)

    # Recentre: subtract duration-weighted mean so sum_r dur_r * D_r == 0.
    weighted_mean = (durations[:, None, None, None] * raw).sum(axis=0) / T
    D = raw - weighted_mean[None]                       # (n_regimes, *base.shape)

    for r, (lo, hi) in enumerate(bounds):
        A_t[lo:hi] = base + D[r]
    return A_t


# --------------------------------------------------------------------------- #
# Time-varying latent simulation                                              #
# --------------------------------------------------------------------------- #
def simulate_from_tensor(
    A_t: np.ndarray,
    noise_scale: float = 1.0,
    burn_in: int = 50,
    seed: int | None = None,
) -> np.ndarray:
    """Simulate latents (T, d) from a per-timepoint adjacency A_t.
    During burn-in the first-regime adjacency A_t[0] is used."""
    T, p, d, _ = A_t.shape
    rng = np.random.default_rng(seed)
    total = burn_in + T
    x = np.zeros((total + p, d), dtype=np.float64)
    noise = noise_scale * rng.standard_normal((total + p, d))
    # Precompute B_t = A_t[l].T per regime-step lazily inside the loop.
    for t in range(p, total + p):
        step = t - p
        A_step = A_t[0] if step < burn_in else A_t[step - burn_in]
        acc = np.zeros(d)
        for l in range(p):
            acc += A_step[l].T @ x[t - (l + 1)]
        x[t] = acc + noise[t]
    return x[p + burn_in:]


# --------------------------------------------------------------------------- #
# Combined subject generator (used by Part 6)                                  #
# --------------------------------------------------------------------------- #
@dataclass
class SubjectSim:
    x: np.ndarray          # (T, d) latents
    A_t: np.ndarray        # (T, max_lag, d, d) effective coupling over time
    site_base: np.ndarray  # (max_lag, d, d) the site's static base adjacency


def generate_subject(
    base_dag: DAG,
    site_index: int,
    T: int = 200,
    *,
    cross_site: bool,
    temporal: bool,
    n_edges_shifted: int = 2,
    temporal_strength: float = 0.6,
    site_shift_scale: float = 0.5,
    n_regimes: int = 2,
    noise_scale: float = 1.0,
    burn_in: int = 50,
    base_seed: int = 0,
    seed: int | None = None,
) -> SubjectSim:
    """Generate one subject under the requested variation configuration."""
    site_dag = (make_site_dag(base_dag, site_index, n_edges_shifted,
                              site_shift_scale, base_seed)
                if cross_site else
                make_site_dag(base_dag, site_index, 0))
    # Independent streams for the temporal regimes and the innovation noise so
    # the two are not coupled through a shared seed.
    ss = seed if isinstance(seed, np.random.SeedSequence) \
        else np.random.SeedSequence(seed)
    temp_ss, sim_ss = ss.spawn(2)
    A_t = make_time_varying(
        site_dag, T,
        temporal_strength=temporal_strength if temporal else 0.0,
        n_regimes=n_regimes, seed=temp_ss,
    )
    x = simulate_from_tensor(A_t, noise_scale=noise_scale,
                             burn_in=burn_in, seed=sim_ss)
    return SubjectSim(x=x, A_t=A_t, site_base=site_dag.A)


# --------------------------------------------------------------------------- #
# Independence metrics + mandatory 2x2 test                                    #
# --------------------------------------------------------------------------- #
def temporal_variation(A_t: np.ndarray) -> float:
    """Max absolute deviation of the coupling from its own time-average."""
    return float(np.abs(A_t - A_t.mean(axis=0, keepdims=True)).max())


def _config_ensemble(base_dag, cross_site, temporal, n_sites=4, n_subj=5,
                     T=200, n_edges_shifted=2, temporal_strength=0.6):
    """Return (max temporal variation over subjects, max cross-site difference
    of per-site mean time-averaged mechanism)."""
    site_means = []
    max_tvar = 0.0
    for s in range(n_sites):
        per_subj_timeavg = []
        for k in range(n_subj):
            sim = generate_subject(
                base_dag, s, T,
                cross_site=cross_site, temporal=temporal,
                n_edges_shifted=n_edges_shifted,
                temporal_strength=temporal_strength,
                seed=1000 * s + k,
            )
            max_tvar = max(max_tvar, temporal_variation(sim.A_t))
            per_subj_timeavg.append(sim.A_t.mean(axis=0))
        site_means.append(np.mean(per_subj_timeavg, axis=0))
    site_means = np.stack(site_means)
    cs_diff = 0.0
    for a in range(n_sites):
        for b in range(a + 1, n_sites):
            cs_diff = max(cs_diff, float(np.abs(site_means[a] - site_means[b]).max()))
    return max_tvar, cs_diff


def _selftest() -> None:
    np.set_printoptions(suppress=True)
    print("=" * 72)
    print("PART 3 TEST  -  two variation injectors + INDEPENDENCE")
    print("=" * 72)
    base = generate_dag(d=10, edge_density=1.5, max_lag=2, seed=0)
    TOL = 1e-9

    # ---- sanity: each knob actually does something --------------------------
    s0 = make_site_dag(base, 0, n_edges_shifted=2)
    s1 = make_site_dag(base, 1, n_edges_shifted=2)
    print("\n[knob sanity]")
    print(f"  site0 vs base   differing entries: "
          f"{int((s0.A != base.A).sum())}  (expect ~2)")
    print(f"  site0 vs site1  differing entries: "
          f"{int((s0.A != s1.A).sum())}  (>0 => sites differ)")
    A_t_temp = make_time_varying(base, T=200, temporal_strength=0.6, seed=7)
    print(f"  temporal ON     time-variation: {temporal_variation(A_t_temp):.4f} "
          f"(>0 => regimes differ)")
    A_t_const = make_time_varying(base, T=200, temporal_strength=0.0, seed=7)
    print(f"  temporal OFF    time-variation: {temporal_variation(A_t_const):.2e} "
          f"(==0 => constant)")

    # ---- mandatory directional independence checks --------------------------
    print("\n[CRITICAL TEST 1]  cross-site ON, temporal OFF")
    print("  claim: within each subject, coupling is CONSTANT over time")
    tvar_1, csdiff_1 = _config_ensemble(base, cross_site=True, temporal=False)
    print(f"    max temporal variation (any subject): {tvar_1:.3e}   "
          f"-> {'PASS (no temporal leak)' if tvar_1 < TOL else 'FAIL'}")
    print(f"    cross-site difference present:         {csdiff_1:.4f}   "
          f"-> {'OK, cross-site is active' if csdiff_1 > TOL else 'WARN: no site effect'}")

    print("\n[CRITICAL TEST 2]  temporal ON, cross-site OFF")
    print("  claim: across sites, the base mechanisms are IDENTICAL")
    tvar_2, csdiff_2 = _config_ensemble(base, cross_site=False, temporal=True)
    print(f"    cross-site difference (site means):    {csdiff_2:.3e}   "
          f"-> {'PASS (no cross-site leak)' if csdiff_2 < TOL else 'FAIL'}")
    print(f"    temporal variation present:            {tvar_2:.4f}   "
          f"-> {'OK, temporal is active' if tvar_2 > TOL else 'WARN: no temporal effect'}")

    # ---- full 2x2 independence table ---------------------------------------
    print("\n[2x2 INDEPENDENCE TABLE]  metric == ~0 means 'that axis is OFF'")
    print(f"  {'config':24s} {'temporal_var':>14s} {'crosssite_diff':>16s}")
    grid = [("off / off",       False, False),
            ("cross ON / temp OFF", True,  False),
            ("temp ON / cross OFF", False, True),
            ("both ON",          True,  True)]
    results = {}
    for name, cs, tp in grid:
        tv, cd = _config_ensemble(base, cross_site=cs, temporal=tp)
        results[(cs, tp)] = (tv, cd)
        print(f"  {name:24s} {tv:14.3e} {cd:16.3e}")

    # ---- explicit confirmation + assertions --------------------------------
    print("\n[LEAKAGE VERDICT]")
    no_temp_leak = results[(True, False)][0] < TOL      # cross ON => temporal_var ~0
    no_cs_leak = results[(False, True)][1] < TOL        # temp ON  => crosssite_diff ~0
    cross_works = results[(True, False)][1] > TOL
    temp_works = results[(False, True)][0] > TOL
    print(f"  cross-site does NOT introduce temporal variation : "
          f"{'CONFIRMED' if no_temp_leak else 'VIOLATED'}")
    print(f"  temporal does NOT introduce cross-site difference : "
          f"{'CONFIRMED' if no_cs_leak else 'VIOLATED'}")
    print(f"  (cross-site knob is effective: {cross_works};  "
          f"temporal knob is effective: {temp_works})")

    assert no_temp_leak, "LEAK: temporal variation appeared with temporal OFF"
    assert no_cs_leak, "LEAK: cross-site difference appeared with cross-site OFF"
    assert cross_works and temp_works, "a knob is inert"
    # both-on must show BOTH axes active
    assert results[(True, True)][0] > TOL and results[(True, True)][1] > TOL
    print("\nALL PART 3 INDEPENDENCE CHECKS PASSED")


if __name__ == "__main__":
    _selftest()
