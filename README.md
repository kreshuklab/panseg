# PanSeg  <!-- omit in toc -->


![alt text](panseg/resources/header_h.png)

[![Release new version](https://github.com/kreshuklab/panseg/actions/workflows/release.yml/badge.svg)](https://github.com/kreshuklab/panseg/actions/workflows/release.yml)
[![Run tests](https://github.com/kreshuklab/panseg/actions/workflows/test.yml/badge.svg)](https://github.com/kreshuklab/panseg/actions/workflows/test.yml)
[![Docs Build and Publish](https://github.com/kreshuklab/panseg/actions/workflows/build-and-publish-docs.yml/badge.svg)](https://github.com/kreshuklab/panseg/actions/workflows/build-and-publish-docs.yml)

[![Anaconda-Server Badge](https://anaconda.org/conda-forge/panseg/badges/version.svg)](https://anaconda.org/conda-forge/panseg)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/panseg/badges/latest_release_date.svg)](https://anaconda.org/conda-forge/panseg)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/panseg/badges/downloads.svg)](https://anaconda.org/conda-forge/panseg)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/panseg/badges/license.svg)](https://anaconda.org/conda-forge/panseg)

![Illustration of Pipeline](../assets/images/main_figure_nologo.png)

PanSeg is a tool for cell instance aware segmentation in densely packed 3D volumetric images.
The pipeline uses a two stages segmentation strategy (Neural Network + Segmentation).
The pipeline is tuned for plant cell tissue acquired with confocal and light sheet microscopy.
Pre-trained models are provided.

## Table of Contents  <!-- omit in toc -->

* [Getting Started](#getting-started)
* [Installation](#installation)
* [Repository Index](#repository-index)
* [Citation](#citation)

## Getting Started

Checkout the [**documentation** 📖](https://kreshuklab.github.io/panseg/latest/chapters/getting_started/) for more details.

<https://github.com/user-attachments/assets/9551210b-0ed6-4f06-b1d1-4059aadecd11>

## Installation

The easiest way to get PanSeg is using the installer. [**Download it here**](https://heibox.uni-heidelberg.de/d/72b4bd3ba5f14409bfee/)

The installer comes with python and conda.
Please go to the [documentation](https://kreshuklab.github.io/panseg/latest/chapters/getting_started/installation/) for more detailed instructions.

For development, we recommend to clone the repo and install using:

```bash
conda env create -f environment-dev.yaml
```

The above command will create new conda environment `panseg-dev` together with all required dependencies.

## Repository Index

The PanSeg repository is organised as follows:

* **panseg**: Contains the source code of PanSeg.
* **docs**: Contains the documentation of PanSeg.
* **examples**: Contains the files required to test PanSeg.
* **tests**: Contains automated tests that ensures the PanSeg functionality are not compromised during an update.
* **evaluation**: Contains all script required to reproduce the quantitative evaluation in
[Wolny et al.](https://doi.org/10.7554/eLife.57613).
* **conda-reicpe**: Contains all necessary code and configuration to create the anaconda package.
* **constructor**: Contains scripts for the installer creation.
* **Menu**: Contains scripts for OS Menu entries

## PanSeg & PlantSeg

This project stated 2020 as `PlantSeg`, but its capabilities are not
restricted to plant cells! To better reflect that, for the 2.0 release
we changed the name to `PanSeg`, referring to plant and animal cells.

## Citation

### PanSeg 2.0

A new publication highlighting all the new features is currently on its way.  
Until then, if you find our work useful, please cite the PlantSeg 1.0 paper below.

### Plantseg 1.0

```text
@article{wolny2020accurate,
  title={Accurate and versatile 3D segmentation of plant tissues at cellular resolution},
  author={Wolny, Adrian and Cerrone, Lorenzo and Vijayan, Athul and Tofanelli, Rachele and Barro, Amaya Vilches and Louveaux, Marion and Wenzl, Christian and Strauss, S{\"o}ren and Wilson-S{\'a}nchez, David and Lymbouridou, Rena and others},
  journal={Elife},
  volume={9},
  pages={e57613},
  year={2020},
  publisher={eLife Sciences Publications Limited}
}
```
