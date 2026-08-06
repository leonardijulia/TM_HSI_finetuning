# Hyperspectral Downstream Benchmarking using GFMs 
This repository contains the experimental code for fine-tuning the TerraMind Geospatial Foundation Model (GFM) for Hyperspectral downstream tasks. The codebase is based on [this Hydra template](https://github.com/ashleve/lightning-hydra-template) and the datasets and work of [SpectralEarth](https://github.com/AABNassim/spectral_earth).

The preliminary results of these experiments are summarized in a short paper accepted at the ML4RS workshop at ICLR2026: [Spectral Gaps and Spatial Priors: Studying Hyperspectral Downstream Adaptation Using TerraMind](https://arxiv.org/abs/2603.06690).

### 🔬 Short overview
GFMs are typically trained on RGB or Multispectral Sentinel-2 data. This project investigates strategies to fine-tune these pre-trained backbones for Hyperspectral inputs. Current work was done on TerraMind, but the aim is to expand these experiments to other pretrained GFMs, both HSI-specific and non-HSI-specific. 
Currently, the codebase supports the following downstream datasets:
1. EnMAP-CORINE - *Land Cover Multilabel Classification*
2. EnMAP-BNETD - *Land Cover Segmentation*
3. EnMAP-CDL - *Crop classification*
4. ENMAP-BDForet - *Tree Species Segmentation*
5. Hyperview-1 - *Soil property estimation*
6. Hyperview-2 - *Soil property estimation*
7. **upcoming**  EMIT-CH4 - *Methane detection*

The setup supports two band selection techniques with backbones pretrained on Sentinel-2 data, such as TerraMind: **Naive Band Selection** and **SRF Grouping**. More details on them can be found in our [preprint](https://arxiv.org/abs/2603.06690).

### ⚙️ Stack
Hydra, TerraTorch, Lightning, TorchGeo.

### 🔜 Upcoming developments
- Support for new HSI downstream datasets (such as GHG detection);
- Support for other GFM backbones, including HSI-pretrained ones.
