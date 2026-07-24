"""Part 2 - Latent VAR dynamics for the MERIDIAN fMRI simulator.

Given a lagged DAG adjacency tensor A (Part 1), simulate latent time series by
vector autoregression:

    x[t, j] = sum_{l=0}^{max_lag-1} sum_i A[l, i, j] * x[t-(l+1), i]
              + noise_scale * e[t, j],      e ~ N(0, 1)

In matrix form x[t] = sum_l B_l @ x[t-(l+1)] + e[t] with B_l = A[l].T (so that
output j collects over source i). A burn-in is discarded so transients die out.

Stationarity is checked via the spectral radius of the VAR(p) companion matrix;
if it ever reaches 1 the coupling strengths are shrunk until it is safely below
a target. For a topological-order DAG (Part 1) the B_l are strictly triangular,
so the spectral radius is exactly 0 and no rescaling is needed -- but the check
and rescale path are implemented generally for future structures.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from generate_dag import DAG, generate_dag

ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "figures"


def coefficient_matrices(A: np.ndarray) -> np.ndarray:
    """A (max_lag, d, d) with A[l,i,j]=i->j  ->  B (max_lag, d, d) with
    x[t] = sum_l B_l x[t-(l+1)];  B_l = A[l].T."""
    return np.transpose(A, (0, 2, 1)).copy()


def companion_matrix(A: np.ndarray) -> np.ndarray:
    """Build the (p*d, p*d) VAR(p) companion matrix from adjacency A."""
    B = coefficient_matrices(A)
    p, d, _ = B.shape
    C = np.zeros((p * d, p * d), dtype=np.float64)
    C[:d, :] = np.hstack([B[l] for l in range(p)])     # top block row = [B_1..B_p]
    if p > 1:
        C[d:, :-d] = np.eye((p - 1) * d)               # sub-diagonal identities
    return C


def spectral_radius(A: np.ndarray) -> float:
    eig = np.linalg.eigvals(companion_matrix(A))
    return float(np.max(np.abs(eig)))


def ensure_stationary(
    A: np.ndarray, target: float = 0.95, shrink: float = 0.9, max_iter: int = 50
) -> tuple[np.ndarray, float, int]:
    """Shrink all coupling strengths until spectral radius < target.

    Returns (possibly-rescaled A, final spectral radius, n_rescales).
    """
    A = A.copy()
    rho = spectral_radius(A)
    n = 0
    while rho >= target and n < max_iter:
        A *= shrink
        rho = spectral_radius(A)
        n += 1
    return A, rho, n


def simulate_latents(
    dag: DAG,
    T: int = 200,
    noise_scale: float = 1.0,
    burn_in: int = 50,
    seed: int | None = None,
    enforce_stationary: bool = True,
) -> np.ndarray:
    """Simulate latent time series of shape (T, d) from a DAG."""
    rng = np.random.default_rng(seed)
    A = dag.A
    if enforce_stationary:
        A, _, _ = ensure_stationary(A)
    B = coefficient_matrices(A)
    p, d, _ = B.shape

    total = burn_in + T
    x = np.zeros((total + p, d), dtype=np.float64)   # first p rows are history pad
    noise = noise_scale * rng.standard_normal((total + p, d))
    for t in range(p, total + p):
        acc = np.zeros(d)
        for l in range(p):
            acc += B[l] @ x[t - (l + 1)]
        x[t] = acc + noise[t]

    return x[p + burn_in:]          # drop history pad + burn-in -> (T, d)


def plot_latents(x: np.ndarray, path: str | Path, title: str = "") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T, d = x.shape
    spacing = 4.0 * np.median(x.std(axis=0))
    fig, ax = plt.subplots(figsize=(11, 6))
    for j in range(d):
        ax.plot(x[:, j] + j * spacing, lw=0.8)
        ax.text(-2, j * spacing, f"x{j}", ha="right", va="center", fontsize=8)
    ax.set_xlabel("timepoint")
    ax.set_ylabel("latent (offset for display)")
    ax.set_yticks([])
    ax.set_title(title or f"Latent VAR dynamics  (T={T}, d={d})")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _selftest() -> None:
    print("=" * 70)
    print("PART 2 TEST  -  latent VAR dynamics")
    print("=" * 70)
    dag = generate_dag(d=10, edge_density=1.5, max_lag=2, seed=0)

    rho = spectral_radius(dag.A)
    print(f"companion spectral radius = {rho:.6f}   (stationary if < 1)")
    print(f"  -> strictly-triangular DAG => expected ~0, no rescale needed")

    A_resc, rho_resc, n_resc = ensure_stationary(dag.A)
    print(f"ensure_stationary: {n_resc} rescales, final rho = {rho_resc:.6f}")

    x = simulate_latents(dag, T=200, noise_scale=1.0, burn_in=50, seed=1)
    print(f"\nsimulated latents shape = {x.shape}  (expect (200, 10))")
    print(f"all finite             = {np.isfinite(x).all()}")
    print(f"global max |x|         = {np.abs(x).max():.3f}   (not exploding)")
    print(f"per-latent std         = "
          f"{np.array2string(x.std(axis=0), precision=2)}")
    print(f"min per-latent std     = {x.std(axis=0).min():.3f}   (not dead)")

    # Stationarity sanity: first-half vs second-half variance should be similar
    half = x.shape[0] // 2
    v1 = x[:half].var(axis=0).mean()
    v2 = x[half:].var(axis=0).mean()
    print(f"var first half / second half = {v1:.3f} / {v2:.3f}  "
          f"(ratio {v1 / v2:.2f}, ~1 => stable, no drift)")

    fig_path = FIGURES / "example_latents.png"
    plot_latents(x, fig_path)
    print(f"\nlatent plot saved -> {fig_path}")

    assert x.shape == (200, 10)
    assert np.isfinite(x).all()
    assert np.abs(x).max() < 1e3
    assert x.std(axis=0).min() > 1e-6
    assert rho < 1.0
    print("\nALL PART 2 CHECKS PASSED")


if __name__ == "__main__":
    _selftest()
