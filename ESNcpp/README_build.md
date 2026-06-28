# Closing figures — build & run

Two deliverables:

* **`TFM_closing_figures.ipynb`** — one autonomous notebook that produces **Figure 1**
  (MC across the lifespan + structural predictors) and **Figure 2** (communicability
  decomposition, regime change, PCA), plus supplementary control/PCA/UMAP panels.
* **`cpp/`** — five small C++/Eigen programs that compute memory capacity (and the
  surrogate / biological-input / IPC / single-trace variants), parallelised with OpenMP.

The notebook compiles and drives the C++ for you, caches every result to `procdata/`,
and writes figures to `figures/`. You normally just open it and run top to bottom.

---

## 1 · One-time setup (Windows)

1. **Compiler** — install [MSYS2](https://www.msys2.org/), then in the *MSYS2 UCRT64*
   shell: `pacman -S mingw-w64-ucrt-x86_64-gcc`. This gives
   `C:\msys64\ucrt64\bin\g++.exe` (auto-detected).
2. **Eigen** — download Eigen (3.4+ or master) and place the folder as
   **`eigen-master/`** right next to the `.cpp` files in `cpp/` (so that
   `cpp/eigen-master/Eigen/Dense` exists). No build needed — Eigen is header-only.
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

Put `data_with_metrics.pkl` (one row per connectome; the DataFrame you already have,
with `connectome`, `age`, `dataset`, `Ratio`, `kY_obs`, `C_w`, `comm_mean`,
`strength`, `degree`, `density`, …) under **`procdata/`**.

The notebook adds a stable `sid = arange(len(df))` as the C++ merge key, and computes the
extra decomposition features (diagonal self-communicability, Taylor T2/T3, Zhang–Horvath
clustering) from the connectome matrices.

---

## 3 · Run

Open `TFM_closing_figures.ipynb` and run all cells. The C++ binaries are built on first
run and cached; every experiment is cached to `procdata/*.pkl`. To force a clean rebuild
set `FORCE_RECOMPILE = True` and/or `FORCE_RECOMPUTE = True` in §0.

To compile the programs by hand instead:

```powershell
# Windows
powershell -ExecutionPolicy Bypass -File cpp\build.ps1
```
```bash
# Linux / macOS
make -C cpp EIGEN=/path/to/eigen-master
```

---

## 4 · The five programs (`cpp/`)

All share `esn.hpp` and read a `key=value` `config.txt`; the notebook writes those
configs and the connectome CSVs (`cpp/data/`) automatically.

| program | what it computes | output |
|---|---|---|
| `esn_mc`         | memory capacity for every subject (the workhorse) | `subject_id, MC_Glob` |
| `esn_surrogates` | MC of the real net vs three weight nulls (uniform / reshuffle / broken-stick) over realizations | tidy `subject_id, realization, model, MC` |
| `esn_bio`        | MC with global vs thalamic input vs a random-input-pair null | `subject_id, MC, MC_Bio, MC_BioNull` |
| `esn_ipc`        | information-processing capacity by Legendre order (P1, P2, P1·P1) | `subject_id, Lin, Quad, Cross11` |
| `esn_trace`      | one subject: input, reservoir states, readout, and the MC(τ) curve | `trace_*.csv` |

Conventions (identical across programs and notebook): reservoir update
`r(t)=tanh(Win·u(t)+W·r(t−1))`, connectome diagonal zeroed and rescaled to spectral
radius `rho=0.99`, input `u(t)~U(−1,1)`, ridge readout, and MC(τ)=`r²` on a held-out test
split (Damicelli's squared correlation), with `MC = Σ_τ r²(τ)`.
