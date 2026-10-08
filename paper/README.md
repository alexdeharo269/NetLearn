# paper/ — analyses and figures of the v2 manuscript

Everything here reads the connectomes and the definitive outputs of `chain/chain_analysis.py`
(input scale 1e-5, symmetric surrogates). Nothing in the rest of the repo is modified.

| script | what it does | time |
|---|---|---|
| `paper_analysis.py` | per-subject stable rank, local/global heterogeneity factors, truncated Gramian series, input-Gramian spectrum (effective rank, decay rate), closed-form MC and noise-floor sweep; per-cohort and camCAN descriptives; surrogate summary; writes `data/` and `data/numbers.json` (every number quoted in the text) | ~30 s (+~5 min with `--ridge-sweep`) |
| `make_figures.py` | all figures from `data/` only → `figures/*.pdf, *.png` (one function per figure) | ~20 s |
| `exact_noise_free.py` | noise-free memory in ball arithmetic (needs `pip install python-flint`) → `data/noise_free_exact.csv` | ~10 min |

```bash
# from the repo root, after chain_analysis.py has written chain/results (scales 1 and 1e-5)
python paper/paper_analysis.py --data procdata/data_with_metrics.pkl --chain chain/results --ridge-sweep
python paper/make_figures.py                # or: python paper/make_figures.py fig4_memory
python paper/exact_noise_free.py --data procdata/data_with_metrics.pkl --n 24
```

On the Mac, if the ESN binaries complain about libomp, prefix the first command with
`DYLD_LIBRARY_PATH=$CONDA_PREFIX/lib` as for `chain_analysis.py`.

`cross_species_v2.ipynb` is an exploration, not part of the manuscript: closed-form memory of the
mesoscale connectomes packaged with bio2art (fly, two mouse, human, macaque, marmoset; downloaded to
`~/.cache/bio2art`, not committed) against the human stable-rank line → `figures/explore_cross_species.png`.

Main-text figures: `fig2_order2` (disparity → spectrum), `fig4_memory` (memory counts input
directions above the noise floor), `fig5_lifespan`. Figures 1 and 3 of the manuscript are unchanged.
SI figures: `figS_theory`, `figS_gramian`, `figS_noise`, `figS_energy`, `figS_global`,
`figS_surrogates`, `figS_taylor`, `figS_cohorts`.

The `data/` and `figures/` outputs are committed (with `git add -f`, since `.gitignore` excludes
csv/json/pdf) so that the figures can be edited without the connectomes. `data/work/` is not committed.

## v3 (notebooks)

- `analysis_v3.ipynb`: structure of each connectome with the labels of its parcellation (HCPd, HCPya, camCAN and HCPa
  networks follow the AAL3 order), exact factorisation of the stable rank, strongest edge and homotopic share, null
  models in closed form (including equal strength and interhemispheric x1/2), within-cohort aging and virtual rescaling
  of interhemispheric weights, CALM referred vs comparison children, thresholding vs binarisation, and nodal disparity
  against degree. Set the data paths in its second cell (`all_data.mat` and `demographics.csv` of the original release
  are needed for CALM). Writes `paper/data/v3/`.
- `figures_v3.ipynb`: Figs. 2 and 3 of the manuscript (single column, seaborn, 400 dpi) from `paper/data/` only
  -> `paper/figures/v3/`. Fig. 1 is v1's memory paradigm (`trace_fit.pdf`).
