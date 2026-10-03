# Analytical chain: disparity → subgraph centrality → Gramian → spectrum → memory capacity

Code for the analytical link between local weight disparity and reservoir memory,
on the real, symmetric connectomes and on symmetric weight surrogates (`ESNcpp/esn.hpp`).
The sign-flip null has negative weights; the asymmetric control is directed on purpose.

| step | relation | what the pipeline measures |
|---|---|---|
| 1 | disparity → $C_{ii}$ | $(\tilde W^2)_{ii}=\tilde s_i^2Y_i$; R² of $C_{ii}\approx 1+\tfrac12\tilde s_i^2Y_i$; identity $\sum_i\tilde s_i^2Y_i=\sum_k\lambda_k^2$ |
| 2 | $C_{ii}$ ↔ Gramian | node-level correlation of $C_{ii}$ with $(G_I)_{ii}=[(I-\tilde W^2)^{-1}]_{ii}$ |
| 3 | linear theory ↔ ESN | closed-form MC of the linear reservoir (same $W_{in}$, same ridge per sample, no simulation) against the simulated tanh ESN, per input scale |
| 4 | structure → MC | MC against $\langle\tilde s^2Y\rangle=\sum_k\lambda_k^2/N$, $\langle\lvert\lambda\rvert\rangle$ (Aceituno et al.), $\sum\lambda^3/N$, $\sum\lambda^4/N$, $\langle C_{ii}\rangle$, $C_w$, …; R² increments over $\sum\lambda^2$ |
| regime | development vs aging | r with MC of shape ($R=Y/Y_{null}$), scale ($\langle\tilde s^2\rangle$), $\langle\tilde s^2Y\rangle$, $C_w$; age ≤ 32 vs > 32 |
| nulls | surrogates | ΔMC vs real, Wilcoxon; $\sum\lambda^2/N$ and linear theory per surrogate; the real weights made asymmetric as a control |

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
python chain/chain_analysis.py --data ../NetLearn-data             # full sample, scales 1 and 1e-5
python chain/chain_analysis.py --synthetic 80 --out chain/results_synthetic   # pipeline test, no data
```

Options: `--scales 1 1e-5`, `--subset 500`, `--threads 8`, `--surr-real 5`, `--surr-n 100`,
`--skip-surrogates`, `--skip-esn` (reuse `<out>/work/*.csv`), `--data path/to/data_with_metrics.pkl`
(reads the pickle directly). On a Mac the compiler needs OpenMP: `CXX=g++-14 python chain/...`
(Homebrew GCC) or the conda clang you use for `make`.

The ESN runs call `ESNcpp/esn_mc` and `ESNcpp/esn_surrogates` (built automatically) with the
manuscript hyper-parameters, `RES` + `MAIN` / `SURR` of `ESNcpp/TFM_closing_figures.ipynb`:
ρ = 0.99, ridge = 1e-6 (× input_scale²), train 0.7, washout 1000, 10 000 steps, τ_max = 20, one
input projection per subject, seed 42; surrogates: 5 realizations on the notebook's 100-subject
subset (read from `<data>/cache/mc_surrogates.*`; a random subset otherwise). At input scale 1
the MC is the manuscript MC (reproduction check in the summary). A full run takes ~10 min on 4 cores.

Outputs in `chain/results/` (ignored by git):

| file | content |
|---|---|
| `chain_summary.md` / `.json` | every number of the chain: steps 1–4, R² increments, regime, surrogates, manuscript (directed) nulls from the cache, lifespan, reproduction |
| `chain_subject.csv` | one row per connectome: structural quantities, MC, ⟨r²⟩ and MC_lin per scale, metadata |
| `chain_surrogates_scale*.csv` | surrogate runs: subject, realization, model, MC, Σλ²/N, MC_lin |
| `chain_nodes_sample.csv` | node-level values for 150 random subjects (panels a–b) |
| `fig_chain.png` / `.pdf` | (a) step 1, (b) step 2, (c) theory vs simulation, (d) Σλ²/N → MC, (e) surrogates on the real-network line, (f) regime |

## 3 · Changes to the C++ engine (backward compatible)

Optional config keys (all off by default, so default outputs are unchanged):

* `input_scale` (default `1.0`) multiplies every entry of `W_in ~ U(-1,1)`. The random
  stream does not depend on it, and at `1.0` the output is bit-for-bit identical to the
  previous binary (checked).
* `log_state` (`esn_mc`): adds `r2_mean`, the mean squared reservoir state ⟨r_i(t)²⟩ after the
  washout (the linearity diagnostic).
* `log_theory` (`esn_mc`, `esn_surrogates`): adds `MC_lin`, the memory capacity of the linear
  reservoir x(t) = W̃x(t−1) + W_in u(t) with the same W_in, read out by population ridge
  regression with the simulation's penalty per training sample (ridge / n_train):
  m(τ) = (gᵀMg)² / (σ²gᵀMCMg), g = σ²W̃^τW_in, M = (C + ridge/n_train·I)⁻¹, σ² = 1/3, with the
  state covariance C from the doubling algorithm (`mc_linear_theory` in `esn.hpp`).
  `esn_surrogates` also adds `m2` = Tr W̃²/N = Σλ²/N.
* `asym_control` (`esn_surrogates`): adds `RealAsym`, the real weights made asymmetric (each
  undirected edge keeps its weight one way and takes a random other edge's weight the other
  way). It has its own random stream, so the other models' rows do not change.

**Symmetric surrogates.** Reshuffle and Broken Stick used to be built row by row, which
made them directed. In the linear regime asymmetry alone raises MC (the real weights made
asymmetric gain ≈ +3 MC at input scale 1e-5, +0.5 at 1), so both are now symmetric, on the
real edge set:

* Reshuffle: the real weights are permuted over the undirected edges, then pairs of edges
  swap weights (simulated annealing) until every node is back to its own strength sᵢ and
  Σⱼwᵢⱼ², i.e. its own Yᵢ (median residuals 0.1 % and 0.3 %). Same per-node weight
  distribution, random fibre → weight assignment.
* Broken Stick: iid Exp(1) weights (the broken-stick law once normalised), scaled
  symmetrically to the real strengths, then swapped so each node's Yᵢ matches the
  broken-stick mean 2/(kᵢ+1). Strengths exact; ⟨kY/(2k/(k+1))⟩ ≈ 1.05 (real ≈ 1.5).

On the 100-subject subset, median ΔMC (null − real), input scale 1e-5 [and 1]: Uniform
−1.85 [−0.69], Broken Stick −0.83 [−0.47], Reshuffle +0.07 [+0.06, n.s.], Sign Flip +0.01
[+0.18]. The rescaling to ρ changes the sign-flipped matrix's order-2 term.

**Input scale and ridge.** Every program (`esn_mc`, `esn_surrogates`, `esn_bio`, `esn_ipc`,
`esn_trace`) reads `input_scale`, and multiplies `ridge` by `input_scale²`. In the linear
regime the states scale with the input, so this keeps the regularisation relative; at
`input_scale=1` nothing changes. In the notebook set `INPUT_SCALE` (§0, cell 2) and run with
`FORCE_RECOMPUTE = True` (cached `procdata/*.pkl` hold the old scale-1 results). Note that at
1e-5 the IPC quadratic/cross capacities vanish (the reservoir is linear).
