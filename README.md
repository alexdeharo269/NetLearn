<h1 align="center">Weight disparity in brain connectomes shapes reservoir memory across the human lifespan</h1>

<p align="center">
  <b>Alejandro de Haro García</b><sup>1</sup> &nbsp;·&nbsp; <b>M. Ángeles Serrano</b><sup>1,2,3</sup>
</p>
<p align="center">
  <sup>1</sup>Departament de Física de la Matèria Condensada, Universitat de Barcelona &nbsp;·&nbsp;
  <sup>2</sup>UBICS &nbsp;·&nbsp; <sup>3</sup>ICREA
</p>

<p align="center">
  <img src="assets/approach.svg" width="100%" alt="Animated explanation of the approach in eleven scenes. The connectome is a network of 90 brain regions linked by nerve fibres, and every link has a weight. Disparity measures how evenly a region spreads its strength over its connections; subgraph centrality counts the loops by which activity returns to a region; the stable rank asks whether the weight of the whole network is shared or held by a few connections. The connectome is then used as the fixed recurrent layer of a neural network: a random signal is fed into every region and a trained readout tries to recall the signal a number of steps back. Recall fades with the delay; this is the forgetting curve, and the area under it is the memory capacity. Two connectomes with different weights have different memory capacities.">
</p>

<p align="center"><sub>
The approach of the paper, one idea at a time. Everything drawn is computed from two connectomes of the dataset.
</sub></p>

## The idea

Brain networks are usually studied as wiring diagrams, although diffusion MRI gives a weight to every connection.
We ask what those weights do.

We take human structural connectomes, from birth to ninety years, and use each one as the fixed recurrent layer of an
echo-state network. A random signal drives the network, and a trained readout tries to recover what the signal was a number
of steps back. How well it does, summed over the delays, is the memory capacity of that connectome.

On the structural side we look at how the weights are arranged: the disparity of each region, the loops counted by
subgraph centrality, and the stable rank of the whole network. The paper shows how these three are tied together, how they
predict memory capacity, and how both change across the lifespan.

## Repository

| path | contents |
|---|---|
| [`notebooks/01_data_and_structure.ipynb`](notebooks/01_data_and_structure.ipynb) | builds the dataset from the released connectomes; network statistics, disparity and its null model |
| [`notebooks/02_esn_memory.ipynb`](notebooks/02_esn_memory.ipynb) | compiles and drives the C++ engine: memory capacity of every connectome, surrogates, the single-subject trace |
| [`notebooks/03_paper_v2.ipynb`](notebooks/03_paper_v2.ipynb) | walk expansion, stable rank, statistics, and the figures and numbers of the manuscript |
| [`engine/`](engine/) | C++17/Eigen/OpenMP echo-state engine (`esn.hpp`) and its programs; see [`engine/README.md`](engine/README.md) |
| [`src/`](src/) | `chain_analysis.py` (the chain from weights to memory) and `export_data.py` |
| [`animation/`](animation/) | the animation above, in plain JavaScript |
| [`note/`](note/) | bibliography |

## Reproducing the paper

1. **Data.** Download the structural connectomes and demographics released by Mousley et al. from <https://osf.io/7p4y3/>
   into `preprocdata/` (`all_data.mat`, `demographics.csv`). They are public; this repository does not redistribute them.
2. **Environment.** Python with NumPy, SciPy, pandas, matplotlib and seaborn; a C++17 compiler and Eigen for the engine
   (setup in [`engine/README.md`](engine/README.md)).
3. **Run the notebooks in order**, `01 → 02 → 03`. Notebook 02 builds the engine on first use and caches every run.

## The animation

GitHub does not run scripts in a README, so the animation is an SVG file animated with SMIL. It is written by the
JavaScript in [`animation/`](animation/), which has no dependencies:

| file | role |
|---|---|
| `esn.js` | port of `engine/esn.hpp`: the same random stream, reservoir update and ridge readout |
| `atlas.js` | centroids of the 90 regions and the outline of the brain seen from above |
| `fig_approach.js` | the eleven scenes and their timeline |
| `svg.js`, `build.js` | SVG helpers and the command line |

Nothing in it is drawn by hand. The regions sit at the centroids of their atlas parcels and line width is connection weight
(in the stable-rank scene, the share of the squared weight carried by the fifteen strongest connections).
The signals, the readouts and the forgetting curves are simulated with the settings of the paper; on the first connectome the
JavaScript port and the C++ engine give the same forgetting curve to six decimal places.

To rebuild it, export two connectomes with `export_connectomes` of notebook 02 (it already writes the subject of the paradigm
figure to `engine/data/connectome_one.csv`) and run:

```bash
node animation/build.js --connectome engine/data/connectome_one.csv --contrast engine/data/connectome_two.csv
```

Both must come from HCPd, HCPya, camCAN or HCPa, whose region order is the one of `atlas.js`.

## Data

The connectomes were reconstructed from diffusion MRI by Mousley et al. (*Topological turning points across the human lifespan*, 2025)
in nine datasets: dHCP, BCP, CALM, RED, ACE, HCPd, HCPya, camCAN and HCPa. We use them as released, with raw streamline
counts as weights, and restrict the analysis to the 3,901 neurotypical participants.

## Citation

```bibtex
@unpublished{deharo2026weight,
  title  = {Weight disparity in brain connectomes shapes reservoir memory across the human lifespan},
  author = {de Haro Garc{\'i}a, Alejandro and Serrano, M. {\'A}ngeles},
  year   = {2026},
  note   = {Manuscript in preparation}
}
```

## Acknowledgments

We thank Laia Barjuan for helpful discussions. M. Á. S. acknowledges support from Grant PID2022-137505NB-C22 funded by
MCIN/AEI/10.13039/501100011033 and by ERDF/EU, and Grant PID2025-169261NB-C21 funded by MICIU/AEI/10.13039/501100011033.
