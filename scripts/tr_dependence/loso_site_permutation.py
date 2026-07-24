"""Dependence-robustness for the ABIDE-I TR result: leave-one-site-out + a
site-level permutation null for the Spearman rho between |delta_TR| and DIAGONAL
VAR(1) rejections across the 91 site pairs (published rho=0.956), with the
off-diagonal as the control.

Does NOT reimplement the diagnostic. The per-pair diagonal/off-diagonal rejection
counts are taken from the EXISTING diagnostic output results/test_b_marginal.csv
using the SAME counting as scripts/test_c_tr_instability.py (find the FDR reject
column; is_diag = region_i==region_j; sum per (site_1,site_2)). This script
reproduces the published rho=0.956 as a start-up check, then adds:

(1) LEAVE-ONE-SITE-OUT: drop each of the 14 sites and all pairs involving it,
    recompute Spearman(|delta_TR|, diagonal count) and (|delta_TR|, off-diagonal
    count) on the remaining 78 pairs. Report min/median/max of the 14 rho values
    for each partition.

(2) SITE-LEVEL PERMUTATION: the 91 pairs are dependent (14 sites). Permute the 14
    measured TR values ACROSS THE 14 SITES (respecting site exchangeability),
    recompute each pair's |delta_TR|, recompute Spearman against the FIXED
    rejection counts. n_perm=10000. Two-sided permutation p for the observed
    diagonal rho and the off-diagonal rho, plus the matched-TR vs mismatched-TR
    median diagonal-rejection contrast under the same null.

TR source: results/measured_tr.csv if present (the canonical file, once the TR
work is merged); otherwise an embedded snapshot of those measured values
(provenance below) so this runs standalone. No git operations; new folder only.

PRE-REGISTERED EXPECTATION (report honestly even if it fails):
  diagonal rho stays high (~0.9+) across all 14 LOSO folds and is extreme under
  the site-level null (small permutation p); off-diagonal stays n.s. throughout.

Resumable: LOSO cells are per-site atomic JSON (skip-existing); the permutation
null is checkpointed atomically every chunk and resumes deterministically.
"""
from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent.parent
MARGINAL = ROOT / "results/test_b_marginal.csv"
TR_CANONICAL = ROOT / "results/measured_tr.csv"      # present only after the TR merge
OUT = ROOT / "results/tr_dependence"
LOSO_CELLS = OUT / "_loso_cells"
CKPT = OUT / "_perm_ckpt.npz"

N_PERM = 10000
CHUNK = 1000
SEED = 20260628
PUBLISHED_RHO_DIAG = 0.9556

# Embedded snapshot of the per-site MEASURED TR (s) and retained N, provenance:
# measured from NIfTI headers by the TR step-1 script, cached in
# results/measured_tr.csv (origin/main). Used only if that canonical file is not
# yet in the working tree; identical values.
_TR_SNAPSHOT = {
    "PITT": (1.5, 44), "LEUVEN_2": (1.6519099473953247, 30), "NYU": (2.0, 169),
    "UM_1": (2.0, 81), "UM_2": (2.0, 30), "USM": (2.0, 60), "YALE": (2.0, 46),
    "TRINITY": (2.0, 44), "STANFORD": (2.0, 36), "SDSU": (2.0, 33),
    "KKI": (2.5, 39), "UCLA_1": (3.0, 54), "MAX_MUN": (3.0, 41),
    "CALTECH": (2.0, 35),
}


def load_tr():
    if TR_CANONICAL.exists():
        d = pd.read_csv(TR_CANONICAL)
        print(f"[tr] canonical {TR_CANONICAL.name}")
        return dict(zip(d.site, d.tr_s)), dict(zip(d.site, d.N))
    print("[tr] embedded snapshot (canonical results/measured_tr.csv not present)")
    return ({k: v[0] for k, v in _TR_SNAPSHOT.items()},
            {k: v[1] for k, v in _TR_SNAPSHOT.items()})


def find_reject_col(df):
    cands = [c for c in df.columns if "reject" in c.lower() and "fdr" in c.lower()]
    if not cands:
        cands = [c for c in df.columns if "reject" in c.lower()]
    assert len(cands) == 1, f"ambiguous reject columns: {cands}"
    return cands[0]


def build_pairs(tr):
    """Per-pair diagonal/off-diagonal rejection counts + |delta_TR| -- the exact
    counting used by scripts/test_c_tr_instability.py, on the present marginal
    output."""
    df = pd.read_csv(MARGINAL)
    rc = find_reject_col(df)
    df["is_diag"] = df.region_i == df.region_j
    rows = []
    for (s1, s2), g in df.groupby([df.site_1, df.site_2]):
        rows.append(dict(site_1=s1, site_2=s2,
                         n_diag_reject=int(g.loc[g.is_diag, rc].sum()),
                         n_offdiag_reject=int(g.loc[~g.is_diag, rc].sum())))
    pairs = pd.DataFrame(rows)
    assert len(pairs) == 91, f"expected 91 pairs, got {len(pairs)}"
    pairs["delta_TR"] = [abs(tr[a] - tr[b]) for a, b in
                         zip(pairs.site_1, pairs.site_2)]
    return pairs.sort_values(["site_1", "site_2"]).reset_index(drop=True)


def _spear(x, y):
    r, p = stats.spearmanr(x, y)
    return float(r), float(p)


def _atomic(path, write_fn):
    tmp = Path(str(path) + ".tmp")
    write_fn(tmp)
    tmp.replace(path)


# --------------------------------------------------------------------------- #
# (1) leave-one-site-out
# --------------------------------------------------------------------------- #
def run_loso(pairs, sites):
    LOSO_CELLS.mkdir(parents=True, exist_ok=True)
    rows = []
    for s in sites:
        cell = LOSO_CELLS / f"drop_{s}.json"
        if cell.exists():
            rows.append(json.loads(cell.read_text()))
            continue
        sub = pairs[(pairs.site_1 != s) & (pairs.site_2 != s)]
        rd, pd_ = _spear(sub.delta_TR, sub.n_diag_reject)
        ro, po = _spear(sub.delta_TR, sub.n_offdiag_reject)
        rec = dict(dropped_site=s, n_pairs=int(len(sub)),
                   rho_diag=rd, p_diag=pd_, rho_off=ro, p_off=po)
        _atomic(cell, lambda t, r=rec: Path(t).write_text(json.dumps(r)))
        rows.append(rec)
    return pd.DataFrame(rows).sort_values("dropped_site").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# (2) site-level permutation null
# --------------------------------------------------------------------------- #
def _matched_contrast(delta, ndiag):
    """median diagonal rejections for matched-TR (delta==0) minus mismatched."""
    m = delta == 0
    if m.sum() == 0 or (~m).sum() == 0:
        return np.nan
    return float(np.median(ndiag[m]) - np.median(ndiag[~m]))


def run_permutation(pairs, sites, n_perm=N_PERM):
    OUT.mkdir(parents=True, exist_ok=True)
    s_idx = {s: i for i, s in enumerate(sites)}
    i1 = pairs.site_1.map(s_idx).to_numpy()
    i2 = pairs.site_2.map(s_idx).to_numpy()
    tr_vec = np.array([pairs.attrs["tr"][s] for s in sites])
    ndiag = pairs.n_diag_reject.to_numpy()
    noff = pairs.n_offdiag_reject.to_numpy()

    rng = np.random.default_rng(SEED)
    if CKPT.exists():
        z = np.load(CKPT)
        done = int(z["done"])
        rd_null = list(z["rho_diag"][:done]); ro_null = list(z["rho_off"][:done])
        con_null = list(z["contrast"][:done])
        for _ in range(done):                      # advance rng deterministically
            rng.permutation(len(sites))
        print(f"[perm] resumed from checkpoint at {done}/{n_perm}")
    else:
        done, rd_null, ro_null, con_null = 0, [], [], []

    for i in range(done, n_perm):
        perm = rng.permutation(len(sites))
        tp = tr_vec[perm]
        d = np.abs(tp[i1] - tp[i2])
        rd_null.append(stats.spearmanr(d, ndiag)[0])
        ro_null.append(stats.spearmanr(d, noff)[0])
        con_null.append(_matched_contrast(d, ndiag))
        if (i + 1) % CHUNK == 0 or i + 1 == n_perm:
            _save_ckpt(i + 1, rd_null, ro_null, con_null)
            print(f"[perm] {i+1}/{n_perm}")
    return (np.array(rd_null), np.array(ro_null), np.array(con_null))


def _save_ckpt(done, rd, ro, con):
    # np.savez appends ".npz"; use a tmp that already ends in .npz so it writes
    # exactly there, then atomically rename onto the checkpoint.
    tmp = Path(str(CKPT) + ".tmp.npz")
    np.savez(str(tmp), done=done, rho_diag=np.array(rd),
             rho_off=np.array(ro), contrast=np.array(con))
    tmp.replace(CKPT)


def two_sided_p(null, obs):
    null = null[np.isfinite(null)]
    return float((1 + np.sum(np.abs(null) >= abs(obs))) / (len(null) + 1))


# --------------------------------------------------------------------------- #
def report(pairs, obs, loso, perm, n_perm):
    rd_null, ro_null, con_null = perm
    p_diag = two_sided_p(rd_null, obs["rho_diag"])
    p_off = two_sided_p(ro_null, obs["rho_off"])
    con_obs = _matched_contrast(pairs.delta_TR.to_numpy(), pairs.n_diag_reject.to_numpy())
    p_con = two_sided_p(con_null, con_obs)
    matched = pairs[pairs.delta_TR == 0]; mism = pairs[pairs.delta_TR > 0]

    OUT.mkdir(parents=True, exist_ok=True)
    _atomic(OUT / "loso_permutation.csv",
            lambda t: loso.to_csv(t, index=False))

    def mmm(col):
        v = loso[col]
        return v.min(), v.median(), v.max()
    dmin, dmed, dmax = mmm("rho_diag")
    omin, omed, omax = mmm("rho_off")

    L = [
        "# TR dependence-robustness: leave-one-site-out + site-level permutation",
        "",
        "Per-pair diagonal/off-diagonal rejection counts are the existing "
        "diagnostic output (results/test_b_marginal.csv), counted exactly as "
        "test_c_tr_instability.py. Start-up check reproduces the published "
        f"diagonal Spearman rho: **{obs['rho_diag']:+.4f}** "
        f"(p={obs['p_diag']:.2e}); off-diagonal control **{obs['rho_off']:+.4f}** "
        f"(p={obs['p_off']:.3f}).", "",
        "## (1) Leave-one-site-out (14 folds, 78 pairs each)", "",
        "| dropped site | n pairs | rho diagonal | rho off-diagonal |",
        "|---|---|---|---|",
    ]
    for _, r in loso.iterrows():
        L.append(f"| {r['dropped_site']} | {int(r['n_pairs'])} | "
                 f"{r['rho_diag']:+.4f} | {r['rho_off']:+.4f} |")
    L += [
        "",
        f"**Diagonal rho across folds — min {dmin:+.4f}, median {dmed:+.4f}, "
        f"max {dmax:+.4f}.**",
        f"**Off-diagonal rho across folds — min {omin:+.4f}, median "
        f"{omed:+.4f}, max {omax:+.4f}.**", "",
        f"## (2) Site-level permutation null (n_perm={n_perm}, TRs permuted "
        "across the 14 sites)", "",
        "| quantity | observed | two-sided permutation p |",
        "|---|---|---|",
        f"| diagonal rho | {obs['rho_diag']:+.4f} | {p_diag:.4g} |",
        f"| off-diagonal rho (control) | {obs['rho_off']:+.4f} | {p_off:.4g} |",
        f"| matched vs mismatched median diag contrast | {con_obs:+.1f} | {p_con:.4g} |",
        "",
        f"Matched-TR pairs (delta_TR=0): n={len(matched)}, median diagonal "
        f"rejections={np.median(matched.n_diag_reject):.1f}. Mismatched "
        f"(delta_TR>0): n={len(mism)}, median={np.median(mism.n_diag_reject):.1f}.",
        "",
        "## Verdict", "",
        _verdict(dmin, omin, omax, p_diag, p_off), "",
    ]
    _atomic(OUT / "loso_permutation.md", lambda t: Path(t).write_text("\n".join(L) + "\n"))
    print(f"\nwrote {OUT/'loso_permutation.md'} and .csv")
    print(f"  LOSO diagonal rho min/med/max: {dmin:+.4f}/{dmed:+.4f}/{dmax:+.4f}")
    print(f"  LOSO off-diag  rho min/med/max: {omin:+.4f}/{omed:+.4f}/{omax:+.4f}")
    print(f"  permutation p: diagonal={p_diag:.4g}  off-diagonal={p_off:.4g}  "
          f"contrast={p_con:.4g}")


def _verdict(dmin, omin, omax, p_diag, p_off):
    holds = dmin >= 0.85 and p_diag <= 0.01 and p_off > 0.05 and abs(omin) < 0.6 \
        and abs(omax) < 0.6
    if holds:
        return ("ROBUST: the TR->diagonal correlation survives leave-one-site-out "
                f"(min fold rho {dmin:+.3f}) and is extreme under the site-level "
                f"permutation null (p={p_diag:.4g}), while the off-diagonal "
                f"control stays non-significant (permutation p={p_off:.3g}). The "
                "result is not driven by any single site or by pair "
                "non-independence.")
    return ("DOES NOT fully hold -- inspect the folds/p-values above; report as-is "
            "(a single-site dependence or a significant off-diagonal would be a "
            "real finding).")


def main(smoke=False):
    tr, _N = load_tr()
    pairs = build_pairs(tr)
    pairs.attrs["tr"] = tr
    sites = sorted(set(pairs.site_1) | set(pairs.site_2))
    assert len(sites) == 14, f"expected 14 sites, got {len(sites)}"
    rd, pd_ = _spear(pairs.delta_TR, pairs.n_diag_reject)
    ro, po = _spear(pairs.delta_TR, pairs.n_offdiag_reject)
    obs = dict(rho_diag=rd, p_diag=pd_, rho_off=ro, p_off=po)
    assert abs(rd - PUBLISHED_RHO_DIAG) < 0.01, \
        f"start-up check failed: diagonal rho {rd:.4f} != published {PUBLISHED_RHO_DIAG}"
    print(f"[check] reproduced diagonal rho={rd:+.4f} (published {PUBLISHED_RHO_DIAG}); "
          f"off-diagonal rho={ro:+.4f}")

    loso = run_loso(pairs, sites)
    n_perm = 500 if smoke else N_PERM
    perm = run_permutation(pairs, sites, n_perm=n_perm)
    if smoke:
        rd_null, ro_null, con_null = perm
        print(f"\nSMOKE (LOSO + {n_perm}-perm):")
        print(f"  LOSO diagonal rho min/median/max: "
              f"{loso.rho_diag.min():+.4f}/{loso.rho_diag.median():+.4f}/{loso.rho_diag.max():+.4f}")
        print(f"  LOSO off-diag  rho min/median/max: "
              f"{loso.rho_off.min():+.4f}/{loso.rho_off.median():+.4f}/{loso.rho_off.max():+.4f}")
        print(f"  perm p (n={n_perm}): diag={two_sided_p(rd_null, rd):.4g}  "
              f"off={two_sided_p(ro_null, ro):.4g}")
        print("SMOKE OK (partial permutation; no report written). The 500-perm "
              "checkpoint + LOSO cells are a deterministic PREFIX of the full "
              "run, so the tmux run resumes from them (500->10000). To run all "
              "10000 fresh instead, first delete results/tr_dependence/"
              "_perm_ckpt.npz and _loso_cells/.")
        return
    report(pairs, obs, loso, perm, n_perm)


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
