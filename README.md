<h1 align="center">NetLearn</h1>

<p align="center">
  <b>What do the weights of the brain's wiring do for its memory?</b><br>
  Human connectomes, from birth to ninety, used as recurrent neural networks.
</p>

<p align="center">
  <a href="#what-is-this">What is this</a> &nbsp;·&nbsp;
  <a href="#what-we-find">What we find</a> &nbsp;·&nbsp;
  <a href="#how-it-works">How it works</a> &nbsp;·&nbsp;
  <a href="#getting-started">Getting started</a>
</p>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/approach_dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="assets/approach_light.svg">
  <img src="assets/approach_light.svg" width="100%" alt="Animated explanation of the project in fourteen scenes. The connectome is a network of 90 brain regions linked by nerve fibres, and every link has a weight. Disparity measures how evenly a region spreads its strength over its connections; subgraph centrality adds up the loops by which activity returns to a region; the stable rank counts the independent directions that carry the weight of the whole network. The connectome is then used as the fixed recurrent layer of a neural network: a random signal is fed into every region and a trained readout tries to recall the signal a number of steps back. Recall fades with the delay; this is the forgetting curve, and the area under it is the memory capacity. Two connectomes with different weights have different memory capacities. In the example the heaviest links join the two hemispheres, and halving those weights on the same wiring raises the memory capacity.">
</picture>

<p align="center"><sub>
The whole idea in a minute and a half. Everything drawn is computed from two real connectomes.
</sub></p>

## What is this

The connectome is the map of nerve fibres between the regions of a brain. It is usually studied as a wiring diagram:
which regions are connected to which. The measurements also say how strongly, and that strength is far from even.
A few connections are very strong and most are weak.

NetLearn asks what that arrangement of weights does. It takes each connectome, uses it as the recurrent layer of a
small neural network, and measures how long the network keeps a trace of a signal fed into it.

## What we find

- **Weights matter, not only wiring.** Keep every connection in place, change only how strong each one is, and the
  network remembers a different amount.
- **Shared weight means more memory.** When a few connections hold most of the weight, activity has few independent
  directions to travel along and the trace of the input fades quickly. When the weight is shared, the network
  remembers further back.
- **Memory changes with age.** It is highest in newborns, lowest in adulthood, and rises again in old age.
- **The two hemispheres are behind it.** That course follows the weight between the hemispheres, which sits on a
  handful of very strong connections.

Memory here means how long a network retains its input. It is not a measure of what a person remembers.

## How it works

1. **Read the weights.** For every connectome we measure how each region spreads its strength over its connections
   (disparity), how activity loops back to a region (subgraph centrality), and how many independent directions carry the
   weight of the whole network (stable rank).
2. **Turn it into a network.** The connectome becomes the recurrent layer of an echo-state network. Regions are its
   units and fibres are its connections. Nothing in it is trained.
3. **Test its memory.** A random signal drives the network. A linear readout, the only trained part, tries to recover
   what the signal was one step back, two steps back, and so on. Recall fades with the delay, and the area under that
   forgetting curve is the memory capacity.
4. **Change the weights.** Null models keep the wiring and rearrange or rescale the weights, to see which of their
   features memory depends on.

## What's in the repository

| path | contents |
|---|---|
| [`notebooks/01_data_and_structure.ipynb`](notebooks/01_data_and_structure.ipynb) | builds the dataset and measures the structure of every connectome |
| [`notebooks/02_esn_memory.ipynb`](notebooks/02_esn_memory.ipynb) | runs every connectome as an echo-state network and measures its memory; null models |
| [`notebooks/03_paper_v2.ipynb`](notebooks/03_paper_v2.ipynb) | relates structure to memory: statistics and figures |
| [`engine/`](engine/) | the echo-state engine in C++ (Eigen, OpenMP); see [`engine/README.md`](engine/README.md) |
| [`src/`](src/) | Python analysis code |
| [`animation/`](animation/) | the animation above, in plain JavaScript |
| [`note/`](note/) | bibliography |

## Getting started

1. **Get the data.** Download the connectomes and demographics from <https://osf.io/7p4y3/> into `preprocdata/`
   (`all_data.mat`, `demographics.csv`). They are public; this repository does not redistribute them.
2. **Set up Python.**

   ```bash
   pip install numpy scipy pandas matplotlib seaborn networkx nibabel nilearn pymatreader joblib statsmodels
   ```

3. **Set up the engine.** It needs a C++17 compiler and [Eigen](https://eigen.tuxfamily.org/). Notebook 02 builds it
   on first use; [`engine/README.md`](engine/README.md) has the details.
4. **Run the notebooks in order:** `01`, then `02`, then `03`. Every expensive step is cached.

## The animation

GitHub does not run scripts in a README, so the animation is an SVG file animated with SMIL, in a light and a dark
version. It is written by the JavaScript in [`animation/`](animation/), which has no dependencies and includes a port of
the C++ engine.

Nothing in it is drawn by hand. The regions sit at the centroids of their atlas parcels, line width is connection
weight, and every signal, readout and forgetting curve is simulated on the two connectomes shown.

To rebuild it, export two connectomes with `export_connectomes` of notebook 02 and run:

```bash
node animation/build.js --connectome engine/data/connectome_one.csv --contrast engine/data/connectome_two.csv
```

Both must come from HCPd, HCPya, camCAN or HCPa, whose region order is the one of `animation/atlas.js`.

## Data

The connectomes were reconstructed from diffusion MRI and released by Mousley et al.
(*Topological turning points across the human lifespan*, 2025). They cover nine datasets, from newborns to ninety-year-olds:
dHCP, BCP, CALM, RED, ACE, HCPd, HCPya, camCAN and HCPa.

---

<p align="center"><sub>
Alejandro de Haro García and M. Ángeles Serrano · Universitat de Barcelona
</sub></p>
