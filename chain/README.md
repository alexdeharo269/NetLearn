# Analytical chain: disparity → subgraph centrality → Gramian → memory capacity

Code for the analytical link between local weight disparity and reservoir memory,
evaluated on the **base model only** (the real, symmetric connectomes). The weight
surrogates are directed or sign-changed, so the walk expansion does not apply to them;
they stay as empirical tests in the manuscript.

| step | relation | what the pipeline measures |
|---|---|---|
| 1 | disparity → $C_{ii}$ | $(\tilde W^2)_{ii}=\tilde s_i^2Y_i$; R² of $C_{ii}\approx 1+\tfrac12\tilde s_i^2Y_i$; identity $\sum_i\tilde s_i^2Y_i=\sum_k\lambda_k^2$ |
| 2 | $C_{ii}$ ↔ Gramian | node-level correlation of $C_{ii}$ with $(G_I)_{ii}=[(I-\tilde W^2)^{-1}]_{ii}$ |
| 3 | structure → MC | subject-level correlation of MC with $\langle\tilde s^2Y\rangle=\sum_k\lambda_k^2/N$, $\sum_k\lambda_k^2(1-\lambda_k^{2\tau})$, $\langle C_{ii}\rangle$, $\langle\lvert\lambda\rvert\rangle$, … |
| 4 | linear → tanh | MC at the manuscript input scale (1) and near-linear (0.1); mean squared state $\langle r^2\rangle$ |

## 1 · Getting the data here (run on your machine)

The repository is **public**, so the connectomes must not be committed to it.
Push them to a separate **private** repository instead.

```bash
# from the NetLearn repo root, with the environment that reads your pickle
python chain/export_data.py --pkl procdata/data_with_metrics.pkl --out ../NetLearn-data
```

This writes `connectomes_partXX.npz` (exact values, each part < 90 MB), `metadata.csv`
(all scalar columns + `sid`), `cache/` (the cached notebook results from
`ESNcpp/procdata/*.pkl`, used to check that MC is reproduced) and `manifest.json`.

```bash
cd ../NetLearn-data
git init -b main && git add . && git commit -m "Connectome export for the analytical chain"
# create an EMPTY private repo named NetLearn-data on github.com (no README), then:
git remote add origin https://github.com/alexdeharo269/NetLearn-data.git
git push -u origin main
```

Make sure the Claude GitHub App has access to the new repo
(https://claude.ai/connect-github → install / add the repository). Then it can be
attached to the cloud session.

## 2 · Running the chain

```bash
pip install numpy scipy pandas matplotlib
sudo apt install libeigen3-dev        # or: brew install eigen  (Eigen 3.4)
python chain/chain_analysis.py --data ../NetLearn-data             # full sample, scales 1 and 0.1
python chain/chain_analysis.py --synthetic 80 --out chain/results_synthetic   # pipeline test, no data
```

Options: `--scales 1 1e-5`, `--subset 500`, `--threads 8`, `--skip-esn` (reuse
`<out>/work/mc_scale*.csv`), `--data path/to/data_with_metrics.pkl` (reads the pickle directly).

The ESN runs call `ESNcpp/esn_mc` (built automatically) with the manuscript
hyper-parameters, i.e. `RES` + `MAIN` of `ESNcpp/TFM_closing_figures.ipynb`:
ρ = 0.99, ridge = 1e-6, train 0.7, washout 1000, 10 000 steps, τ_max = 20, one
input projection per subject, seed 42. At input scale 1 the MC is the manuscript MC;
if `<data>/cache/mc_main.*` exists, the summary reports the reproduction check.
A full run (~4 000 subjects, two scales) takes a few minutes on 4 cores.

Outputs in `chain/results/` (ignored by git):

| file | content |
|---|---|
| `chain_summary.md` / `.json` | all numbers of the chain (steps 1–4, lifespan check, reproduction check) |
| `chain_subject.csv` | one row per connectome: structural chain quantities, MC and ⟨r²⟩ per scale, metadata |
| `chain_nodes_sample.csv` | node-level values for 150 random subjects (panels a–b) |
| `fig_chain.png` / `.pdf` | (a) step 1, (b) step 2, (c) structure → MC, (d) MC across input scales, (e) lifespan |

## 3 · Changes to the C++ engine (backward compatible)

`ESNcpp/esn.hpp` and `ESNcpp/esn_mc.cpp` gained two optional config keys:

* `input_scale` (default `1.0`) multiplies every entry of `W_in ~ U(-1,1)`. The random
  stream does not depend on it, and at `1.0` the output is bit-for-bit identical to the
  previous binary (checked).
* `log_state` (default `0`): when `1`, `esn_mc` adds an `r2_mean` column with the mean
  squared reservoir state ⟨r_i(t)²⟩ after the washout (the linearity diagnostic).

The comments on the surrogates were also corrected: H0 is the symmetric endpoint mean,
Reshuffle and Broken Stick are directed, and the rescaling to ρ changes the
sign-flipped matrix's order-2 term.

**Input scale and ridge.** Every program (`esn_mc`, `esn_surrogates`, `esn_bio`, `esn_ipc`,
`esn_trace`) reads `input_scale`, and multiplies `ridge` by `input_scale²`. In the linear
regime the states scale with the input, so this keeps the regularisation relative; at
`input_scale=1` nothing changes. In the notebook set `INPUT_SCALE` (§0, cell 2) and run with
`FORCE_RECOMPUTE = True` (cached `procdata/*.pkl` hold the old scale-1 results). Note that at
1e-5 the IPC quadratic/cross capacities vanish (the reservoir is linear).
