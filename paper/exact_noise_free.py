#!/usr/bin/env python3
"""
Noise-free memory of the linear reservoir, computed exactly (SI Note 2.6).

For symmetric W~ = V Lambda V' and eta -> 0 the input vector cancels and
    m(tau) = phi_tau' K^-1 phi_tau,   phi_tau,k = lambda_k^tau,   K_kl = 1 / (1 - lambda_k lambda_l).
K is so ill-conditioned (~2000 bits needed) that double precision is useless, so the solve is
done in ball arithmetic (python-flint, `pip install python-flint`) at two precisions; the
difference between them bounds the error.

Usage (repo root):
  python paper/exact_noise_free.py --data procdata/data_with_metrics.pkl --n 24
writes paper/data/noise_free_exact.csv (one row per connectome: m(0), sum_{tau=1..20} m(tau), ...).
"""
import argparse, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from flint import arb, arb_mat, ctx

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "chain"))
import chain_analysis as ca  # noqa: E402


def memory_curve_exact(lam, taus, bits):
    ctx.prec = bits
    L = [arb(repr(float(x))) for x in lam]
    n = len(L)
    K = arb_mat(n, n, [1 / (1 - L[k] * L[l]) for k in range(n) for l in range(n)])
    B = arb_mat(n, len(taus), [L[k] ** t for k in range(n) for t in taus])
    X = K.solve(B)
    out = []
    for j in range(len(taus)):
        s = arb(0)
        for k in range(n): s += B[k, j] * X[k, j]
        out.append(s)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=24, help="number of connectomes, evenly spaced in the data order")
    ap.add_argument("--bits", type=int, nargs=2, default=[2500, 4000])
    ap.add_argument("--out", default=str(REPO / "paper" / "data" / "noise_free_exact.csv"))
    args = ap.parse_args()

    rho = 0.99   # N = 2 check: lambda = (rho, -rho) gives m(tau) = rho^(4 floor(tau/2)) (1 - rho^4)
    v = memory_curve_exact([rho, -rho], [0, 1, 2, 3], 256)
    print("N=2 check:", [round(float(x.mid()), 8) for x in v],
          "expected", [round(x, 8) for x in [1 - rho**4, 1 - rho**4, rho**4 * (1 - rho**4), rho**4 * (1 - rho**4)]])

    sid, mats, meta = ca.load_pickle(args.data) if str(args.data).endswith(".pkl") else ca.load_export(args.data)
    pick = np.linspace(0, len(sid) - 1, args.n).astype(int)
    taus = list(range(0, 21))
    rows, t0 = [], time.time()
    for i in pick:
        l = np.linalg.eigvalsh(mats[i]); l = rho * l / np.abs(l).max()
        lo, hi = memory_curve_exact(l, taus, args.bits[0]), memory_curve_exact(l, taus, args.bits[1])
        m = [float(x.mid()) for x in hi]
        rows.append(dict(sid=int(sid[i]), m0=m[0], MC_1_20=sum(m[1:]), min_m_1_20=min(m[1:]),
                         max_precision_diff=max(abs(float(a.mid()) - float(b.mid())) for a, b in zip(lo, hi)),
                         max_ball_radius=max(float(x.rad()) for x in hi)))
        print(f"  sid {sid[i]}: sum_(1..20) m = {rows[-1]['MC_1_20']:.12f}", flush=True)
    r = pd.DataFrame(rows)
    if "age" in meta: r = r.merge(meta[["sid", "age"]], on="sid", how="left")
    r.to_csv(args.out, index=False)
    print(f"{len(r)} connectomes in {time.time() - t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
