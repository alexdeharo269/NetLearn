# Weight disparity shapes reservoir memory across the human lifespan

Code accompanying the manuscript *"Weight disparity shapes reservoir memory
across the human lifespan"* (A. de Haro García, advisor M. Á. Serrano,
Universitat de Barcelona).

We treat individual structural connectomes spanning ages 0–90 as fixed
echo-state reservoirs, retaining raw streamline weights, and measure their
linear memory capacity (MC). The central methodological result is that the
**subgraph centrality** `C_ii = (e^W)_ii` decomposes analytically into two
local terms — the **weight disparity** Υ (order 2) and the **weighted
clustering** Cᵂ (order 3) — and that this decomposition explains reservoir
memory and its U-shaped lifespan trajectory.

<!-- Optional: drop the two main figures here once committed.
<p align="center">
  <img src="figures/fig1_distributions.png" width="48%" alt="Cohort-invariant connectome statistics"/>
  <img src="figures/fig2_grid.png" width="48%" alt="Lifespan reorganisation of weight disparity"/>
</p>

**Left** — node-level weight/strength/degree statistics collapse across the
nine lifespan cohorts. **Right** — lifespan reorganisation of weight
disparity and its structural correlates (disparity ratio, subgraph
centrality, weighted clustering).
-->

## What the code does

- Computes node- and subject-level weighted-network metrics (disparity,
  strength, degree, clustering) and the walk expansion of the subgraph
  centrality on spectrally normalised connectomes (ρ = 0.99).
- Runs the echo-state-network memory-capacity experiment, with the
  connectome as a fixed reservoir and only a linear ridge readout trained,
  via a C++/Eigen/OpenMP engine called from Python.
- Builds the four edge-level weight surrogates (uniform, broken-stick,
  reshuffle, sign-flip) used to isolate which aspect of the weight
  distribution supports memory.
- Embeds the structural triple {Υ, Cᵂ, C_ii} (PCA and UMAP) with MC held
  out, and regenerates every figure and table in the manuscript.

## Repository layout

```
.
├── TFM_closing_figures.ipynb   # main analysis + all figures/tables
├── procdata/                   # derived per-subject metrics & MC values
├── data/                       # connectome CSVs exported for the C++ engine
├── figures/                    # generated figures
└── README.md
```

## Data

This project introduces no new data collection. The structural connectomes
and demographic metadata are from Mousley et al. and are available at
<https://osf.io/7p4y3/>. The subset preprocessed by F. C. Yeh is also
distributed through the DSI Studio Fiber Data Hub
(<https://brain.labsolver.org/>).

Raw connectomes are **not** redistributed here — obtain them from the source
above and place `data_with_metrics.pkl` under `procdata/`. The small derived
data products needed to reproduce the figures (per-subject nodal metrics and
MC values) are included, so the figures can be regenerated without rerunning
the full reservoir experiments.

## Requirements

- Python 3.10+ with NumPy, SciPy, pandas, matplotlib, `umap-learn` (v0.5.5)
- A C++ compiler with Eigen and OpenMP (for the ESN engine; the notebook
  autodetects the toolchain, with manual overrides documented inline)


## Citation

If you use this code, please cite the manuscript and the original data source
(Mousley et al.).

