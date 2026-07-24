"""Part 8 - Unified loader for real ABIDE and simulated datasets.

ONE loader for both real and simulated data: `load_dataset(path)` returns the
same `BOLDDataset` object regardless of source, so model-training code has a
single data path.

Time-series key normalisation: simulated files store the BOLD under ``ts``
(finalized format); the real ABIDE file (`abide_harmonized.npz`) stores it
under ``X``. The loader reads whichever is present and exposes it uniformly as
``BOLDDataset.ts``. Both share ``site_ids``, ``subject_ids``, ``T``,
``n_regions``.

    >>> from data_loader import load_dataset
    >>> real = load_dataset(".../abide_harmonized.npz")
    >>> sim  = load_dataset(".../sim_sparse_crosssite_d10_seed0.npz")
    >>> real.ts.shape, sim.ts.shape          # (742,116,200) (1000,200,50)
    >>> sim.is_simulated, sim.ground_truth["ground_truth_dags"].shape
    >>> real.is_simulated, real.ground_truth # False, None
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
REAL_DEFAULT = ROOT.parent / "data" / "processed" / "abide_harmonized.npz"
SIM_DIR = ROOT / "data" / "simulated"

_TS_KEYS = ("ts", "X")                      # accepted time-series key names
_CONFIG_KEYS = ["config_name", "cross_site", "temporal", "n_edges_shifted",
                "sparsity_level", "temporal_strength", "seed"]
_GT_KEYS = ["ground_truth_dags", "base_A", "base_topo_order",
            "mixing_W", "latents"]
_PARAM_KEYS = ["d", "n_regions", "T", "max_lag", "TR", "hrf_time_length",
               "region_specific_hrf", "n_drift_components",
               "snr_target", "thermal_frac"]


def _scalar(v):
    return v.item() if getattr(v, "ndim", None) == 0 else v


@dataclass
class BOLDDataset:
    """Uniform container for real or simulated BOLD data."""
    ts: np.ndarray               # (subjects, T, regions)
    site_ids: np.ndarray         # (subjects,)
    subject_ids: np.ndarray      # (subjects,)
    T: int
    n_regions: int
    is_simulated: bool
    config: dict | None          # sim: config metadata;      real: None
    ground_truth: dict | None    # sim: per-site DAGs + extras; real: None
    params: dict | None          # sim: simulation parameters; real: None
    meta: dict                   # real: phenotype fields;     sim: {}
    source: str

    @property
    def n_subjects(self) -> int:
        return self.ts.shape[0]

    @property
    def sites(self) -> list[str]:
        return sorted(np.unique(self.site_ids).tolist())

    def __repr__(self) -> str:
        kind = "SIM" if self.is_simulated else "REAL"
        cfg = f", config={self.config['config_name']}" if self.is_simulated else ""
        return (f"BOLDDataset[{kind}] ts={self.ts.shape} "
                f"sites={len(self.sites)} subjects={self.n_subjects}{cfg}")


def load_dataset(path: str | Path) -> BOLDDataset:
    """Load a real or simulated dataset into a uniform BOLDDataset."""
    path = Path(path)
    f = np.load(path, allow_pickle=False)
    keys = set(f.files)

    ts_key = next((k for k in _TS_KEYS if k in keys), None)
    if ts_key is None:
        raise KeyError(f"{path.name}: no time-series key {_TS_KEYS}; "
                       f"found {sorted(keys)}")
    for req in ("site_ids", "subject_ids", "T", "n_regions"):
        if req not in keys:
            raise KeyError(f"{path.name}: missing required key '{req}'")

    is_sim = "config_name" in keys
    if is_sim:
        config = {k: _scalar(f[k]) for k in _CONFIG_KEYS if k in keys}
        ground_truth = {k: f[k] for k in _GT_KEYS if k in keys}
        params = {k: _scalar(f[k]) for k in _PARAM_KEYS if k in keys}
        meta: dict = {}
    else:
        config = ground_truth = params = None
        common = set(_TS_KEYS) | {"site_ids", "subject_ids", "T", "n_regions"}
        meta = {k: f[k] for k in keys - common}

    return BOLDDataset(
        ts=f[ts_key], site_ids=f["site_ids"], subject_ids=f["subject_ids"],
        T=int(f["T"]), n_regions=int(f["n_regions"]),
        is_simulated=is_sim, config=config, ground_truth=ground_truth,
        params=params, meta=meta, source=str(path),
    )


def list_simulated(sim_dir: str | Path = SIM_DIR) -> list[Path]:
    return sorted(Path(sim_dir).glob("sim_*.npz"))


def _selftest() -> None:
    print("=" * 74)
    print("PART 8  -  unified loader + finalized-format verification")
    print("=" * 74)

    # ---- one loader, one object type, real vs simulated ----
    print("\n[1] uniform interface (same object for real and simulated)")
    real = load_dataset(REAL_DEFAULT)
    sim = load_dataset(SIM_DIR / "sim_sparse_crosssite_d10_seed0.npz")
    for ds in (real, sim):
        assert isinstance(ds, BOLDDataset)
        _ = (ds.ts, ds.site_ids, ds.subject_ids, ds.T, ds.n_regions,
             ds.n_subjects, ds.sites, ds.is_simulated)   # identical access
    assert real.ts.ndim == sim.ts.ndim == 3
    assert real.ts.shape[1] == real.T and sim.ts.shape[1] == sim.T
    assert real.ts.shape[2] == real.n_regions and sim.ts.shape[2] == sim.n_regions
    print(f"    real: {real!r}")
    print(f"          ground_truth={real.ground_truth}  params={real.params}")
    print(f"          meta keys={sorted(real.meta)}")
    print(f"    sim : {sim!r}")
    print(f"    both are {type(real).__name__}; ts is 3-D (subjects,T,regions) "
          f"for both: OK")

    # ---- required contents present in each simulated file ----
    print("\n[2] finalized required contents per simulated file")
    required_ts = "ts"
    files = list_simulated()
    for p in files:
        raw = set(np.load(p).files)
        assert required_ts in raw, f"{p.name}: on-disk key must be 'ts'"
        assert {"site_ids", "subject_ids"} <= raw
        assert {"config_name", "cross_site", "temporal", "sparsity_level"} <= raw
        assert "ground_truth_dags" in raw
        assert {"d", "n_regions", "TR", "hrf_time_length",
                "snr_target", "thermal_frac", "n_drift_components"} <= raw
    print(f"    all {len(files)} files contain: ts, site_ids, subject_ids,")
    print(f"      config metadata (temporal/cross_site/sparsity_level),")
    print(f"      ground_truth_dags, and sim params (d,N,TR,HRF,noise) -> OK")

    # ---- all 5 configs, each with per-site ground-truth DAGs ----
    print("\n[3] config coverage + ground-truth DAGs")
    by_config: dict[str, list] = {}
    for p in files:
        ds = load_dataset(p)
        by_config.setdefault(ds.config["config_name"], []).append(ds)
    expected = {"sparse_crosssite", "temporal_only", "crosssite_only",
                "both_on", "dense_crosssite"}
    assert set(by_config) == expected, set(by_config)
    print(f"    all 5 configs present: {sorted(by_config)}")
    hdr = ("config", "seeds", "cross", "temp", "sparsity", "gt_dags", "params")
    print(f"    {hdr[0]:17s}{hdr[1]:>9s}{hdr[2]:>7s}{hdr[3]:>6s}"
          f"{hdr[4]:>9s}{hdr[5]:>18s}{hdr[6]:>10s}")
    for name in sorted(by_config):
        dss = by_config[name]
        d0 = dss[0]
        seeds = sorted(d.config["seed"] for d in dss)
        gt = d0.ground_truth["ground_truth_dags"].shape
        print(f"    {name:17s}{str(seeds):>9s}"
              f"{str(bool(d0.config['cross_site'])):>7s}"
              f"{str(bool(d0.config['temporal'])):>6s}"
              f"{d0.config['sparsity_level']:>9s}{str(tuple(gt)):>18s}"
              f"{'TR=' + str(d0.params['TR']):>10s}")

    # ---- verify ground-truth DAG semantics match config ----
    print("\n[4] ground-truth DAG sanity vs config")
    for name in ("temporal_only", "sparse_crosssite", "dense_crosssite"):
        gt = by_config[name][0].ground_truth["ground_truth_dags"]
        spread = float(np.abs(gt - gt[0]).max())     # cross-site spread
        print(f"    {name:17s} per-site DAG spread = {spread:.4f}  "
              f"({'sites identical' if spread == 0 else 'sites differ'})")

    # ---- on-disk summary ----
    print("\n[5] on-disk summary")
    print(f"    {'file':46s}{'ts shape':>18s}{'MB':>8s}")
    real_mb = REAL_DEFAULT.stat().st_size / 1e6
    print(f"    {'REAL abide_harmonized.npz':46s}{str(real.ts.shape):>18s}"
          f"{real_mb:8.1f}")
    total_sim = 0.0
    for p in files:
        ds = load_dataset(p)
        mb = p.stat().st_size / 1e6
        total_sim += mb
        print(f"    {p.name:46s}{str(ds.ts.shape):>18s}{mb:8.1f}")
    print(f"\n    simulated : {len(files)} files, {total_sim:.1f} MB")
    print(f"    real      : 1 file, {real_mb:.1f} MB")
    print(f"    TOTAL on disk (sim + real): {total_sim + real_mb:.1f} MB")

    print("\n    loader works uniformly for real + all simulated: CONFIRMED")
    print("ALL PART 8 CHECKS PASSED")


if __name__ == "__main__":
    _selftest()
