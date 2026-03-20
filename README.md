# PlantSeg  <!-- omit in toc -->

[!Caution]
PlantSeg was renamed to PanSeg to highlight our segmentation capabilities
beyond plants!  
[Please head over to the new repository](https://github.com/kreshuklab/panseg)
The old version will still work, but not receive updates or fixes any more.


![alt text](docs/logos/logo.png)

[![Conda Build and Test](https://github.com/kreshuklab/plant-seg/actions/workflows/build-and-test-package.yml/badge.svg)](https://github.com/kreshuklab/plant-seg/actions/workflows/build-and-test-package.yml)
[![Docs Build and Publish](https://github.com/kreshuklab/plant-seg/actions/workflows/build-and-publish-docs.yml/badge.svg)](https://github.com/kreshuklab/plant-seg/actions/workflows/build-and-publish-docs.yml)

[![Anaconda-Server Badge](https://anaconda.org/conda-forge/plant-seg/badges/version.svg)](https://anaconda.org/conda-forge/plant-seg)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/plant-seg/badges/latest_release_date.svg)](https://anaconda.org/conda-forge/plant-seg)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/plant-seg/badges/downloads.svg)](https://anaconda.org/conda-forge/plant-seg)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/plant-seg/badges/license.svg)](https://anaconda.org/conda-forge/plant-seg)

![Illustration of Pipeline](../assets/images/main_figure_nologo.png)

[PlantSeg](plantseg) is a tool for cell instance aware segmentation in densely packed 3D volumetric images.
The pipeline uses a two stages segmentation strategy (Neural Network + Segmentation).
The pipeline is tuned for plant cell tissue acquired with confocal and light sheet microscopy.
Pre-trained models are provided.

## Table of Contents  <!-- omit in toc -->

* [Getting Started](#getting-started)
* [Installation](#installation)
* [Repository Index](#repository-index)
* [Citation](#citation)

## Getting Started

Checkout the [**documentation** 📖](https://kreshuklab.github.io/plant-seg/latest/chapters/getting_started/) for more details.

https://github.com/user-attachments/assets/9551210b-0ed6-4f06-b1d1-4059aadecd11


## Installation

The easiest way to get PlantSeg is using the installer. [**Download it here**](https://heibox.uni-heidelberg.de/d/72b4bd3ba5f14409bfee/)

The installer comes with python and conda.
Please go to the [documentation](https://kreshuklab.github.io/plant-seg/latest/chapters/getting_started/installation/) for more detailed instructions.

For development, we recommend to clone the repo and install using:

```bash
conda env create -f environment-dev.yaml
```

The above command will create new conda environment `plant-seg-dev` together with all required dependencies.

## Repository Index

The PlantSeg repository is organised as follows:

* **plantseg**: Contains the source code of PlantSeg.
* **docs**: Contains the documentation of PlantSeg.
* **examples**: Contains the files required to test PlantSeg.
* **tests**: Contains automated tests that ensures the PlantSeg functionality are not compromised during an update.
* **evaluation**: Contains all script required to reproduce the quantitative evaluation in
[Wolny et al.](https://doi.org/10.7554/eLife.57613).
* **conda-reicpe**: Contains all necessary code and configuration to create the anaconda package.
* **constructor**: Contains scripts for the installer creation.
* **Menu**: Contains scripts for OS Menu entries

## Citation

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
