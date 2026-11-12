<h1 align="center">Weight disparity in brain connectomes shapes reservoir memory across the human lifespan</h1>

<p align="center">
  <b>Alejandro de Haro García</b><sup>1</sup> &nbsp;·&nbsp; <b>M. Ángeles Serrano</b><sup>1,2,3</sup>
</p>
<p align="center">
  <sup>1</sup>Departament de Física de la Matèria Condensada, Universitat de Barcelona &nbsp;·&nbsp;
  <sup>2</sup>UBICS &nbsp;·&nbsp; <sup>3</sup>ICREA
</p>
<p align="center">
  <a href="#the-result-in-one-figure">Result</a> &nbsp;·&nbsp;
  <a href="#why-the-stable-rank">Mechanism</a> &nbsp;·&nbsp;
  <a href="#repository">Code</a> &nbsp;·&nbsp;
  <a href="#reproducing-the-paper">Reproduce</a> &nbsp;·&nbsp;
  <a href="https://osf.io/7p4y3/">Data</a> &nbsp;·&nbsp;
  <a href="#citation">Cite</a>
</p>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/memory_paradigm_dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="assets/memory_paradigm_light.svg">
  <img src="assets/memory_paradigm_light.svg" width="100%" alt="Animated memory-capacity paradigm. A random scalar input drives a human connectome used as the fixed recurrent layer of an echo-state network; a ridge readout reconstructs the input tau steps in the past; the squared correlation on held-out data as a function of the delay is the forgetting curve, and its sum is the memory capacity. Making the weights around every region uniform lowers the stable rank and the memory capacity.">
</picture>

<p align="center"><sub>
<b>The memory-capacity paradigm (Fig. 4 of the paper), animated.</b>
Everything shown is simulated with the settings of the paper on one HCP Young Adult connectome (27 y, the participant closest to the median age of the sample):
stable rank 3.34, MC = 8.08; with the weights around every region made uniform on the same wiring, stable rank 2.52, MC = 7.61.
The network is drawn with a force-directed layout mirrored across the hemispheres (not anatomical positions); line width is proportional to the weight.
</sub></p>

## In short

Brain networks are usually studied as binary wiring diagrams, although diffusion MRI gives a weight for every connection.
We use 3,901 human structural connectomes (ages 0–90, nine cohorts) as the fixed recurrent layer of an echo-state network and
measure how much past input each one retains.

- **Local weights set the leading term of subgraph centrality.** The Taylor expansion of $`C_{ii} = (e^{\mathbf W})_{ii}`$ couples
  the strength of a region to the disparity of its weights. This order-2 term explains 99.93% of the variance of $`C_{ii}`$
  across all regions of all connectomes.
- **Summed over the network, that term is the stable rank**, and the stable rank predicts memory capacity:
  $`r = 0.95`$ between MC and $`\log r_s`$, which is 89% of the variance.
- **The weights act on memory through the stable rank.** Making the weights around each region uniform lowers MC by 1.89
  (median); reshuffling them while restoring the strength and disparity of every region leaves it nearly unchanged (+0.05).
- **Memory is U-shaped across the lifespan**, with a minimum in the early thirties, and follows the course of the stable rank.
- **Across the lifespan, memory follows the weight between the hemispheres**, which in childhood and adulthood is
  concentrated on a few strong connections.

## The result in one figure

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/lifespan_dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="assets/lifespan_light.svg">
  <img src="assets/lifespan_light.svg" width="100%" alt="Three animated scatter plots of 3,901 connectomes from nine cohorts, added cohort by cohort: stable rank against age, memory capacity against stable rank (r = 0.95 with the logarithm of the stable rank) and memory capacity against age (r = -0.72 up to 32 years and +0.44 after).">
</picture>

<p align="center"><sub>
<b>Stable rank and memory capacity of the 3,901 connectomes</b> (cf. Figs. 3f, 5a and 6a of the paper).
Each point is one connectome. Black curves in <b>a</b> and <b>c</b>: 5-year running mean of the mean at each age, whose minimum is at 32 years
in both panels. Black line in <b>b</b>: least-squares fit on log <i>r</i><sub>s</sub>. Before adulthood the trajectory compares different cohorts.
</sub></p>

| | newborns | 6–12 years | 30–40 years | 76 years or more |
|---|:---:|:---:|:---:|:---:|
| cohorts | dHCP | HCPd | HCPya, camCAN, HCPa | camCAN, HCPa |
| disparity ratio $`\langle\Upsilon/\Upsilon_{\rm null}\rangle`$ | 1.82 | 1.54 | 1.48 | 1.48 |
| stable rank $`r_s`$ | 7.2 | 5.4 | 4.6 | 5.9 |
| memory capacity MC | 11.5 | 10.4 | 9.6 | 10.7 |
| squared weight between hemispheres | 18% | 64% | 65% | 18% |

Values are averaged within each cohort and then over cohorts. MC ranges from 6.7 to 12.4 across the sample (median 10.3).
It correlates with age at $`r = -0.72`$ up to 32 years and $`r = +0.44`$ after, and age adds nothing once the stable rank is known
(partial $`r = -0.06`$ and $`-0.05`$).

## Why the stable rank

Memory in a recurrent network comes from input-driven perturbations that travel along sequences of connections and return.
In the linearised dynamics these paths are the powers of the weighted adjacency matrix, and subgraph centrality collects
the closed ones. With $`\mathbf W`$ rescaled to spectral radius 0.99, its walk expansion is

```math
C_{ii} = \left(e^{\mathbf W}\right)_{ii}
       = 1 + \tfrac12\, s_i^2\, Y_i + \tfrac16\, s_i^2\, c_i^{\rm w}\,(1 - Y_i) + \mathcal O\!\left(\lVert \mathbf W\rVert^4\right),
```

where $`s_i = \sum_j w_{ij}`$ is the strength of region $`i`$, $`Y_i = \sum_j (w_{ij}/s_i)^2`$ the disparity of its weights and
$`c_i^{\rm w}`$ its weighted clustering coefficient. The order-2 term is exact and local. Summed over regions it becomes a
property of the whole network,

```math
\sum_i s_i^2\, Y_i \;=\; \sum_{i,j} w_{ij}^2 \;=\; \sum_k \lambda_k^2 \;=\; \rho(\mathbf W)^2\, r_s ,
\qquad
r_s = \frac{\lVert \mathbf W\rVert_F^2}{\rho(\mathbf W)^2} .
```

The stable rank $`r_s`$ compares the total squared weight with the part carried by the leading eigenvalue. It is small when one
pattern of connectivity carries most of the squared weight and large when the weight is spread over many patterns.
Because every connectome is rescaled to the same spectral radius, a dominant eigenvalue shrinks all the others, and fewer
patterns of connectivity retain the input.

Null models that keep the binary topology test this on an age-balanced sample of 100 connectomes. Three act on the weights
around each region and two on the weights between the hemispheres:

| null model | what it does to the weights | ΔMC (median) | predicted from $`r_s`$ alone |
|---|---|:---:|:---:|
| Uniform | every weight replaced by the mean weight of its two endpoints | −1.89 | −1.94 |
| Broken stick | strengths kept, disparity of each region set to its null expectation | −0.79 | −1.17 |
| Reshuffle | weights permuted, strength and disparity of every region restored | +0.05 | +0.08 |
| Interhemispheric halved | weights between the hemispheres divided by two | +1.07 | +1.05 |
| Interhemispheric equalised | their share of the total weight set to the sample mean (0.29) | +0.32 | +0.29 |

Between the hemispheres the weight sits on very few connections: the one between the left and right supplementary motor areas
alone carries 27% of the squared weight at 30–40 years and is the strongest connection in 90% of those connectomes.
Equalising the interhemispheric share removes most of the rise of memory with age within the aging cohorts
(the correlation of MC with age falls from 0.36 to 0.09 in camCAN and from 0.22 to 0.09 in HCPa).

## The memory test

Each connectome $`\mathbf W`$ (90 regions of the AAL atlas, raw streamline counts) is placed in the recurrent layer of an
echo-state network after rescaling it to $`\rho(\mathbf W) = 0.99`$:

```math
\mathbf r(t+1) = \tanh\!\left[\mathbf W_{\rm in}\, u(t+1) + \mathbf W\, \mathbf r(t)\right],
\qquad
{\rm MC} = \sum_{\tau=1}^{20} r^2(\tau) .
```

The input $`u(t) \sim \mathcal U(-1, 1)`$ is i.i.d. and enters every region through fixed random weights of scale $`10^{-5}`$,
so the reservoir works in its linear regime. Neither the connectome nor the input projection is trained. For each delay
$`\tau`$ a ridge readout reconstructs $`u(t-\tau)`$ from the present state, and $`r^2(\tau)`$ is its squared Pearson correlation
with the target on held-out data ($`T = 10{,}000`$ steps, 1,000 of washout, 70/30 split, ridge $`10^{-6}`$ relative to the input scale).
MC is a measure of short-term retention by the reservoir, not of the cognitive memory of a person.

## Repository

| path | contents |
|---|---|
| [`notebooks/01_data_and_structure.ipynb`](notebooks/01_data_and_structure.ipynb) | builds the dataset from the released connectomes; degree-resolved statistics, disparity and its null model |
| [`notebooks/02_esn_memory.ipynb`](notebooks/02_esn_memory.ipynb) | compiles and drives the C++ engine: MC of every connectome, surrogates, the single-subject trace of the paradigm figure |
| [`notebooks/03_paper_v2.ipynb`](notebooks/03_paper_v2.ipynb) | walk expansion, stable rank, statistics and the figures and numbers of the manuscript |
| [`engine/`](engine/) | header-only C++17/Eigen/OpenMP echo-state engine (`esn.hpp`) and its five programs; see [`engine/README.md`](engine/README.md) |
| [`src/`](src/) | `chain_analysis.py` (the analytical chain from weights to memory) and `export_data.py` |
| [`animation/`](animation/) | the two animated figures above, in plain JavaScript |
| [`note/`](note/) | bibliography |

## Reproducing the paper

1. **Data.** Download the structural connectomes and demographics released by Mousley et al. from <https://osf.io/7p4y3/>
   into `preprocdata/` (`all_data.mat`, `demographics.csv`). They are public; this repository does not redistribute them.
2. **Environment.** Python with NumPy, SciPy, pandas, matplotlib and seaborn; a C++17 compiler and Eigen for the engine
   (setup in [`engine/README.md`](engine/README.md)).
3. **Run the notebooks in order**, `01 → 02 → 03`. Notebook 02 builds the engine on first use and caches every run.

To build the engine by hand:

```bash
make -C engine EIGEN=/path/to/eigen
```

## The animations

GitHub does not run scripts in a README, so the figures are SVG files animated with SMIL and written by a few
dependency-free JavaScript files in [`animation/`](animation/):

| file | role |
|---|---|
| `esn.js` | port of `engine/esn.hpp`: the same Mersenne Twister stream, reservoir update, ridge readout and Uniform null |
| `population.js` | stable rank and MC of every connectome, in parallel |
| `fig_memory.js`, `fig_lifespan.js` | lay out the two figures and their timelines |
| `build.js` | command line |

The JavaScript port is checked against the C++ engine. On the connectome of the first figure the two forgetting curves
agree to $`6 \times 10^{-7}`$ in $`r^2(\tau)`$, and the full run over the 3,901 connectomes gives the numbers of the manuscript:
MC from 6.74 to 12.39 with median 10.25, $`r = 0.946`$ with $`\log r_s`$, and $`r = -0.715`$ and $`+0.441`$ with age before and after 32 years.

Notebook 02 writes the inputs. With them in place:

```bash
node animation/build.js --connectome engine/data/connectome_one.csv --label "90 regions · HCP Young Adult participant, 27 y" --check engine/trace_mccurve.csv
```

```bash
node animation/population.js engine/data/connectomes_all.csv metadata.csv population.csv
```

```bash
node animation/build.js --population population.csv
```

`metadata.csv` is any table with the columns `sid`, `age` and `dataset` (`src/export_data.py` writes one).
The second command takes a few minutes; Node 18 or later is enough and there is nothing to install.

## Data

The connectomes were reconstructed from diffusion MRI by Mousley et al. (*Topological turning points across the human lifespan*, 2025)
in nine datasets: dHCP, BCP, CALM, RED, ACE, HCPd, HCPya, camCAN and HCPa. We use them as released, with raw streamline
counts as weights and neither harmonisation nor density thresholding, and restrict the analysis to the 3,901 neurotypical participants.

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
