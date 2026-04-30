# Installation

!!! note "PlantSeg v1 Documentation"
    To view the installation guide for PlantSeg v1, switch to an older version in the top bar.

## Download

Download the installer from [heibox](https://heibox.uni-heidelberg.de/d/72b4bd3ba5f14409bfee/) and choose according to your platform.

[:material-download: Download](https://heibox.uni-heidelberg.de/d/72b4bd3ba5f14409bfee/){ .md-button .md-button--primary target="_blank"}

## Installation

The installer comes complete with its own python installation. During the installation further dependencies need to be downloaded.

!!! warning "First Start"

    The first start can be quite slow, as the machine learning models need to be downloaded.

=== "Linux and MacOs"

    Download the installer, then open a terminal and run the installer:

    ```bash
    bash PanSeg*.sh
    ```

    On linux, you can start PanSeg through your start menu.

    To start PanSeg in the terminal, navigate to your installation directory
    (default `~/panseg/`) and run `bin/panseg --napari`.

    You might want to add a link to this file to some directory on your $PATH.

    Alternatively, you can activate the panseg conda environment:
    (Replace [INSTALLATION DIR] with the absolute(!) path to your installation)

    ```bash
    eval "$("[INSTALLATION DIR]/bin/conda" shell.bash activate "[INSTALLATION DIR]")
    panseg --help
    ```

===  "Windows"

    Download the installer and execute it. As the installer is not signed you
    might need to confirm the download and that you want to run the installer.

    Choose a installation path without spaces in it, as those can cause issues
    with conda packages.

    Start PlanSeg through the Windows start menu.

=== "conda-forge"

    If you want to install PanSeg without the installer, you need to have
    conda. (We recommend Microforge to get conda,
    see [installing mamba](contributing.md#install-mamba))

    ```bash
    conda create --name panseg panseg
    conda activate panseg
    panseg --napari
    ```

=== "latest git version"

    To get the latest pre-release features, install PanSeg from git. You need to have conda and git installed. (We recommend Microforge to get conda, see [installing mamba](contributing.md#install-mamba))

    ```bash
    git clone https://github.com/kreshuklab/panseg.git
    cd panseg
    conda env create -f environment.yaml
    conda activate panseg
    panseg --napari
    ```

    For development, we recommend using the `environment-dev.yaml` instead!
    Also check [the contributing section](./contributing.md)

## Updating

If you have installed PanSeg *via* the installer or from Conda-forge, you can update to a new version right in the GUI!  
Make sure to start PanSeg from the startmenu, or to activate the conda environment manually (see installation above).

Go to the `Plugins` menu on top, then click `Update Panseg`!

If you have cloned the git repository, you need to update your local repo:

```bash
cd panseg
git pull
conda env update -f environment.yaml # environment-dev.yaml for development
```

## Optional dependencies

Certain compressed TIFF files (e.g., Zlib, ZSTD, LZMA formats) require additional codecs to be processed correctly by PanSeg. To handle such files, install the `imagecodecs` package:

```bash
conda activate panseg
pip install imagecodecs
```

If you plan to use SimpleITK-based watershed segmentation, you will need to install `SimpleITK` as an additional dependency:

```bash
conda activate panseg
pip install SimpleITK
```
