# ESN engine — build & run

Two deliverables:

* **`notebooks/02_esn_memory.ipynb`** — one autonomous notebook that produces **Figure 1**
  (MC across the lifespan + structural predictors) and **Figure 2** (communicability
  decomposition, regime change, PCA), plus supplementary control/PCA/UMAP panels.
* **`engine/`** (this folder) — five small C++/Eigen programs that compute memory capacity (and the
  surrogate / biological-input / IPC / single-trace variants), parallelised with OpenMP.

The notebook compiles and drives the C++ for you, caches every result to `procdata/`,
and writes figures to `figures/`. You normally just open it and run top to bottom.

---

## 1 · One-time setup (Windows)

1. **Compiler** — install [MSYS2](https://www.msys2.org/), then in the *MSYS2 UCRT64*
   shell: `pacman -S mingw-w64-ucrt-x86_64-gcc`. This gives
   `C:\msys64\ucrt64\bin\g++.exe` (auto-detected).
2. **Eigen** — download Eigen (3.4+ or master) and place the folder as
   **`eigen-master/`** right next to the `.cpp` files in `engine/` (so that
   `engine/eigen-master/Eigen/Dense` exists). No build needed — Eigen is header-only.
3. **Python** — any scientific Python (numpy, pandas, scipy, matplotlib, seaborn).
   Optional, for two supplementary panels: `pip install networkx umap-learn`.

If autodetection fails, point the notebook/scripts at your toolchain:

```python
import os
os.environ["GPP"]           = r"C:\msys64\ucrt64\bin\g++.exe"
os.environ["EIGEN_INCLUDE"] = r"C:\path\to\eigen-master"
```

> **Important:** the programs must be compiled with **`-std=c++17`**. Eigen's `LDLT`
> does not compile under C++20 on recent g++, so do not bump the standard. The provided
> build script and the notebook already use C++17.

---

## 2 · Data

Put `data_with_metrics.pkl` (written by `notebooks/01_data_and_structure.ipynb`) (one row per connectome; the DataFrame you already have,
with `connectome`, `age`, `dataset`, `Ratio`, `kY_obs`, `C_w`, `comm_mean`,
`strength`, `degree`, `density`, …) under **`procdata/`** at the repository root.

The notebook adds a stable `sid = arange(len(df))` as the C++ merge key, and computes the
extra decomposition features (diagonal self-communicability, Taylor T2/T3, Zhang–Horvath
clustering) from the connectome matrices.

---

## 3 · Run

Open `notebooks/02_esn_memory.ipynb` and run all cells (it works inside `engine/`). The C++ binaries are built on first
run and cached; every experiment is cached to `engine/procdata/*.pkl`. To force a clean rebuild
set `FORCE_RECOMPILE = True` and/or `FORCE_RECOMPUTE = True` in §0.

To compile the programs by hand instead:

```powershell
# Windows
powershell -ExecutionPolicy Bypass -File engine\build.ps1
```
```bash
# Linux / macOS
make -C engine EIGEN=/path/to/eigen-master
```

---

## 4 · The five programs

All share `esn.hpp` and read a `key=value` `config.txt`; the notebook writes those
configs and the connectome CSVs (`engine/data/`) automatically.

| program | what it computes | output |
|---|---|---|
| `esn_mc`         | memory capacity for every subject (the workhorse) | `subject_id, MC_Glob` |
| `esn_surrogates` | MC of the real net vs four symmetric weight nulls (uniform / broken-stick / reshuffle / sign-flip) over realizations | tidy `subject_id, realization, model, MC` |
| `esn_bio`        | MC with global vs thalamic input vs a random-input-pair null | `subject_id, MC, MC_Bio, MC_BioNull` |
| `esn_ipc`        | information-processing capacity by Legendre order (P1, P2, P1·P1) | `subject_id, Lin, Quad, Cross11` |
| `esn_trace`      | one subject: input, reservoir states, readout, and the MC(τ) curve | `trace_*.csv` |

Conventions (identical across programs and notebook): reservoir update
`r(t)=tanh(Win·u(t)+W·r(t−1))`, connectome diagonal zeroed and rescaled to spectral
radius `rho=0.99`, input `u(t)~U(−1,1)`, ridge readout, and MC(τ)=`r²` on a held-out test
split (Damicelli's squared correlation), with `MC = Σ_τ r²(τ)`.

---

## 5 · Optional config keys

All are off by default, so a config without them gives the same output as before they
existed (checked bit for bit for `esn_mc`, `esn_bio`, `esn_ipc` and `esn_trace`).

* `input_scale` (default `1.0`, every program) multiplies every entry of `Win ~ U(-1,1)`;
  `ridge` is multiplied by `input_scale²` so the regularisation stays relative. The random
  stream does not depend on it.
* `log_state` (`esn_mc`) adds `r2_mean`, the mean squared reservoir state after the washout
  (the linearity diagnostic).
* `log_theory` (`esn_mc`, `esn_surrogates`) adds `MC_lin`, the closed-form memory capacity of
  the linear reservoir with the same `Win` and ridge per training sample
  (`mc_linear_theory` in `esn.hpp`); `esn_surrogates` also adds `m2 = Tr W̃²/N = Σλ²/N`.
* `asym_control` (`esn_surrogates`) adds `RealAsym`, the real weights made asymmetric. It has
  its own random stream, so the other models' rows do not change.

**Surrogates are symmetric.** Reshuffle and Broken Stick keep the real edge set and are
symmetric (weight swaps by simulated annealing back to each node's strength and `Y_i`), because
asymmetry alone raises MC in the linear regime. The earlier row-by-row (directed) versions
gave different numbers, so surrogate results computed before this change are not reproduced
by the current `esn_surrogates`.
