"""Part 1 - Ground-truth DAG generator for the MERIDIAN fMRI simulator.

A latent causal structure is represented as a lagged adjacency tensor

    A[l, i, j] = coupling strength of the lag-(l+1) edge  latent_i -> latent_j

with l in {0, ..., max_lag-1}. There are NO contemporaneous (lag-0) edges, so
the contemporaneous graph is trivially acyclic. We additionally enforce a
random topological order on the *summary* graph (collapse over lags): an edge
i -> j may exist only if i precedes j in that order. This guarantees the
summary graph is a true DAG (no feedback loops, no self-loops) and, as a
bonus, makes the resulting VAR unconditionally stationary (the lagged
coefficient matrices are strictly triangular -> det(I - sum_l B_l z^{l+1}) = 1
for all z, so the companion spectral radius is 0).

The dynamics convention used downstream (Part 2) is

    x[t, j] = sum_{l} sum_{i} A[l, i, j] * x[t-(l+1), i] + innovation.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "figures"


@dataclass
class DAG:
    """A ground-truth lagged causal DAG over ``d`` latent nodes."""
    A: np.ndarray          # (max_lag, d, d) float; A[l,i,j] = lag-(l+1) i->j
    topo_order: np.ndarray # (d,) node indices, source -> sink
    d: int
    max_lag: int

    def summary_adjacency(self) -> np.ndarray:
        """Binary (d, d) graph: 1 where an edge exists at any lag."""
        return (np.abs(self.A).sum(axis=0) > 0).astype(int)

    def n_edges(self) -> int:
        """Number of nonzero lagged edges (counts each (lag,i,j) entry)."""
        return int((np.abs(self.A) > 0).sum())

    def edge_list(self) -> list[tuple[int, int, int, float]]:
        """List of (lag, src, dst, strength) for every nonzero edge."""
        edges = []
        for l in range(self.max_lag):
            for i in range(self.d):
                for j in range(self.d):
                    if self.A[l, i, j] != 0:
                        edges.append((l + 1, i, j, float(self.A[l, i, j])))
        return edges


def _is_acyclic(summary: np.ndarray) -> bool:
    """Kahn's algorithm on the binary summary adjacency (src x dst)."""
    d = summary.shape[0]
    indeg = summary.sum(axis=0).astype(int)        # in-degree per node
    queue = [n for n in range(d) if indeg[n] == 0]
    visited = 0
    while queue:
        n = queue.pop()
        visited += 1
        for m in range(d):
            if summary[n, m]:
                indeg[m] -= 1
                if indeg[m] == 0:
                    queue.append(m)
    return visited == d


def generate_dag(
    d: int,
    edge_density: float = 1.5,
    max_lag: int = 2,
    strength_range: tuple[float, float] = (0.3, 0.7),
    seed: int | None = None,
) -> DAG:
    """Generate a random lagged causal DAG.

    Parameters
    ----------
    d : number of latent nodes.
    edge_density : expected edges per node (total edges ~= density * d).
    max_lag : edges may take an integer lag in {1, ..., max_lag}.
    strength_range : |coupling| sampled uniformly here; sign is random.
    seed : RNG seed for reproducibility.
    """
    rng = np.random.default_rng(seed)

    # Random topological order; an edge i->j is allowed iff rank[i] < rank[j].
    topo_order = rng.permutation(d)
    rank = np.empty(d, dtype=int)
    rank[topo_order] = np.arange(d)

    candidates = [(i, j) for i in range(d) for j in range(d)
                  if rank[i] < rank[j]]
    n_target = int(round(edge_density * d))
    n_edges = min(n_target, len(candidates))

    chosen_idx = rng.choice(len(candidates), size=n_edges, replace=False)
    A = np.zeros((max_lag, d, d), dtype=np.float64)
    for idx in chosen_idx:
        i, j = candidates[idx]
        lag = int(rng.integers(1, max_lag + 1))
        mag = rng.uniform(*strength_range)
        sign = rng.choice([-1.0, 1.0])
        A[lag - 1, i, j] = sign * mag

    dag = DAG(A=A, topo_order=topo_order, d=d, max_lag=max_lag)
    assert _is_acyclic(dag.summary_adjacency()), "generated graph is cyclic!"
    return dag


def is_acyclic(dag: DAG) -> bool:
    return _is_acyclic(dag.summary_adjacency())


def save_dag(dag: DAG, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, A=dag.A, topo_order=dag.topo_order,
        d=np.int64(dag.d), max_lag=np.int64(dag.max_lag),
    )


def load_dag(path: str | Path) -> DAG:
    f = np.load(path)
    return DAG(A=f["A"], topo_order=f["topo_order"],
               d=int(f["d"]), max_lag=int(f["max_lag"]))


def plot_dag(dag: DAG, path: str | Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vmax = float(np.abs(dag.A).max()) or 1.0
    fig, axes = plt.subplots(1, dag.max_lag, figsize=(5 * dag.max_lag, 4.5),
                             squeeze=False)
    for l in range(dag.max_lag):
        ax = axes[0, l]
        im = ax.imshow(dag.A[l], cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(f"lag {l + 1}: A[{l}]  (src i -> dst j)")
        ax.set_xlabel("dst latent j")
        ax.set_ylabel("src latent i")
        ax.set_xticks(range(dag.d))
        ax.set_yticks(range(dag.d))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="strength")
    fig.suptitle(f"Ground-truth DAG  (d={dag.d}, {dag.n_edges()} edges, "
                 f"topo order: {dag.topo_order.tolist()})")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _selftest() -> None:
    print("=" * 70)
    print("PART 1 TEST  -  ground-truth DAG generator")
    print("=" * 70)
    dag = generate_dag(d=10, edge_density=1.5, max_lag=2, seed=0)

    print(f"d           = {dag.d}")
    print(f"max_lag     = {dag.max_lag}")
    print(f"topo order  = {dag.topo_order.tolist()}  (source -> sink)")
    print(f"n_edges     = {dag.n_edges()}  (target ~= {int(1.5 * dag.d)})")
    print(f"acyclic     = {is_acyclic(dag)}")
    density = dag.n_edges() / (dag.d * dag.d)
    print(f"fill        = {density:.3f}  of {dag.d * dag.d} possible entries")

    print("\nEdge list  (lag: src -> dst   strength):")
    for lag, i, j, s in dag.edge_list():
        print(f"  lag {lag}:  {i:2d} -> {j:2d}   {s:+.3f}")

    print("\nAdjacency tensor per lag:")
    for l in range(dag.max_lag):
        print(f"\n-- lag {l + 1}  (rows=src i, cols=dst j) --")
        with np.printoptions(precision=2, suppress=True, linewidth=120):
            print(dag.A[l])

    # round-trip save/load
    tmp = ROOT / "data" / "_part1_selftest_dag.npz"
    save_dag(dag, tmp)
    dag2 = load_dag(tmp)
    roundtrip_ok = np.allclose(dag.A, dag2.A) and \
        np.array_equal(dag.topo_order, dag2.topo_order)
    print(f"\nsave/load round-trip identical: {roundtrip_ok}")
    tmp.unlink()

    fig_path = FIGURES / "example_dag.png"
    plot_dag(dag, fig_path)
    print(f"heatmap saved -> {fig_path}")

    # sanity assertions
    assert is_acyclic(dag)
    assert dag.n_edges() == 15
    assert roundtrip_ok
    print("\nALL PART 1 CHECKS PASSED")


if __name__ == "__main__":
    _selftest()
