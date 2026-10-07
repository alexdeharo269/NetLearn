#!/usr/bin/env python3
"""
Export the analysis DataFrame (data_with_metrics.pkl) to a compact, portable
format that can be pushed to a PRIVATE GitHub repository and read back by
src/chain_analysis.py.

Run it on the machine that has the pickle, from the NetLearn repo root:

    python src/export_data.py --pkl procdata/data_with_metrics.pkl --out ../NetLearn-data

What it writes in --out
    connectomes_partXX.npz  sid (int64), N (int), layout ("upper" or "full"), weights
                            (one row per subject; upper triangle k=1 if every matrix is
                            symmetric, else the full flattened N*N). Values are stored
                            exactly: uint32 if all weights are integers, float64 otherwise.
                            Parts are split so each file stays below --max-mb (GitHub
                            rejects files over 100 MB).
    metadata.csv            every scalar column of the DataFrame + sid
    cache/                  small cached results from engine/procdata (if found), as .pkl
                            and, for DataFrames, also as .csv
    manifest.json           shapes, dtypes, sha256 and library versions

Row order is preserved, so sid = 0..S-1 is the same key notebooks/02_esn_memory.ipynb uses.
"""
import argparse, hashlib, json, platform, shutil, sys
from pathlib import Path

import numpy as np
import pandas as pd


def as_matrix(x):
    A = np.asarray(x, dtype=float)
    if A.ndim == 1:
        n = int(round(np.sqrt(A.size)))
        if n * n != A.size:
            raise ValueError(f"cannot reshape a vector of length {A.size} into a square matrix")
        A = A.reshape(n, n)
    A = A.copy()
    np.fill_diagonal(A, 0.0)                     # same convention as the notebook (_mat)
    return A


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pkl", required=True, help="path to data_with_metrics.pkl")
    ap.add_argument("--out", required=True, help="output folder (e.g. ../NetLearn-data)")
    ap.add_argument("--matrix-col", default="connectome", help="column holding the matrices")
    ap.add_argument("--cache-dir", default=None,
                    help="folder with cached notebook results (default: engine/procdata next to this repo)")
    ap.add_argument("--max-mb", type=float, default=90.0, help="max size of each .npz part")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_pickle(args.pkl).reset_index(drop=True)
    if args.matrix_col not in df.columns:
        sys.exit(f"column '{args.matrix_col}' not found; columns are: {list(df.columns)}")
    S = len(df)
    sid = np.arange(S, dtype=np.int64)
    print(f"{S} rows | columns: {list(df.columns)}")

    mats = [as_matrix(x) for x in df[args.matrix_col]]
    N = mats[0].shape[0]
    bad = [i for i, A in enumerate(mats) if A.shape != (N, N)]
    if bad:
        sys.exit(f"{len(bad)} matrices are not {N}x{N} (first rows: {bad[:5]})")
    n_nan = sum(int(~np.isfinite(A).all()) for A in mats)
    if n_nan:
        print(f"WARNING: {n_nan} matrices contain NaN/inf values (kept as they are)")
    symmetric = all(np.allclose(A, A.T, rtol=0, atol=1e-10) for A in mats)
    integer = all(np.all(A == np.round(A)) and A.min() >= 0 and A.max() < 2**32 for A in mats)
    iu = np.triu_indices(N, k=1)
    layout = "upper" if symmetric else "full"
    dtype = np.uint32 if integer else np.float64
    W = np.stack([(A[iu] if symmetric else A.ravel()) for A in mats]).astype(dtype)
    print(f"N={N} | symmetric={symmetric} -> layout '{layout}' | integer weights={integer} -> {np.dtype(dtype).name}")

    # split into parts that stay below max-mb (estimate from a test compression of 50 rows)
    probe = out / "_probe.npz"
    np.savez_compressed(probe, w=W[: min(50, S)])
    per_row = probe.stat().st_size / min(50, S); probe.unlink()
    rows_per_part = max(1, int(args.max_mb * 1e6 * 0.85 / per_row))
    parts = []
    for p, a in enumerate(range(0, S, rows_per_part)):
        b = min(S, a + rows_per_part)
        f = out / f"connectomes_part{p:02d}.npz"
        np.savez_compressed(f, sid=sid[a:b], weights=W[a:b], N=np.int64(N), layout=np.array(layout))
        parts.append(f)
        print(f"  wrote {f.name}: rows {a}-{b - 1}, {f.stat().st_size / 1e6:.1f} MB")
    too_big = [f.name for f in parts if f.stat().st_size > 99e6]
    if too_big:
        sys.exit(f"parts above 99 MB: {too_big}; re-run with a smaller --max-mb")

    # metadata: every scalar column (array-valued columns are dropped)
    def is_scalar_col(c):
        v = df[c].dropna()
        return len(v) == 0 or not isinstance(v.iloc[0], (np.ndarray, list, tuple, dict))
    keep = [c for c in df.columns if c != args.matrix_col and is_scalar_col(c)]
    dropped = [c for c in df.columns if c != args.matrix_col and c not in keep]
    meta = df[keep].copy(); meta.insert(0, "sid", sid)
    meta.to_csv(out / "metadata.csv", index=False)
    print(f"  wrote metadata.csv: {len(keep)} columns" + (f" (dropped array columns: {dropped})" if dropped else ""))

    # cached notebook results (small files only)
    cache_src = Path(args.cache_dir) if args.cache_dir else Path(__file__).resolve().parents[1] / "engine" / "procdata"
    copied = []
    if cache_src.is_dir():
        (out / "cache").mkdir(exist_ok=True)
        for f in sorted(cache_src.glob("*.pkl")):
            if f.stat().st_size > 25e6:
                continue
            shutil.copy2(f, out / "cache" / f.name); copied.append(f.name)
            try:
                obj = pd.read_pickle(f)
                if isinstance(obj, pd.DataFrame):
                    obj.to_csv(out / "cache" / (f.stem + ".csv"), index=False)
            except Exception:
                pass
        print(f"  copied {len(copied)} cached results from {cache_src}: {copied}")
    else:
        print(f"  (no cache folder at {cache_src}; skipped)")

    manifest = dict(
        n_subjects=S, N=N, layout=layout, dtype=np.dtype(dtype).name, symmetric=symmetric,
        integer_weights=integer, parts=[f.name for f in parts], metadata_columns=keep,
        dropped_columns=dropped, cache=copied,
        sha256={f.name: sha256(f) for f in parts + [out / "metadata.csv"]},
        versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__),
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("  wrote manifest.json\n\nNext: push this folder to a PRIVATE repository (see the docstring of src/export_data.py).")


if __name__ == "__main__":
    main()
