# Contribute to PanSeg

PanSeg is an open-source project, and we welcome contributions from the community. There are many ways to contribute, such as writing tutorials or blog posts, improving the documentation, submitting bug reports and feature requests, or writing code that can be incorporated into PanSeg itself.

## Install Mamba

The easiest way to install PanSeg is by using
[mamba (Miniforge)](https://mamba.readthedocs.io/en/latest/index.html) package manager.
If you don't have conda already, install it:

=== "Linux"

    To download Miniforge open a terminal and type:

    ```bash
    curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
    bash Miniforge3-$(uname)-$(uname -m).sh
    ```

    Then install by typing:

    ```bash
    bash Miniforge3-$(uname)-$(uname -m).sh
    ```

    and follow the installation instructions.
    Please refer to the [Miniforge repo](https://github.com/conda-forge/miniforge) for more information, troubleshooting and uninstallation instructions.
    The miniforge installation file `Miniforge3-*.sh` can be deleted now.

=== "Windows/macOS"

    The first step required to use the pipeline is installing mamba. The installation can be done by downloading the installer from the [Miniforge repo](https://github.com/conda-forge/miniforge). There you can find the download links for the latest version of Miniforge, troubleshooting and uninstallation instructions.

## Getting Started

To set up the development environment, run:

```bash
conda env create -f environment-dev.yaml
conda activate panseg-dev
```

This installs PanSeg using `pip --editable .` and all dependencies using conda. (Some dependencies are only available through conda-forge)

## Hierarchical Design of PanSeg

Please refer to [Python API](../python_api/index.md).

## Coding Style

PanSeg uses _Ruff_ for linting and formatting. _Ruff_ is compatible with _Black_ for formatting.
Commit messages should adhere to conventional commits.

To ensure proper formatting and commit messages, pre-commit is used.
This runs on PRs automatically, to run it locally check out [Pre-commit](https://pre-commit.com/#quick-start).

## Before Submitting a Pull Request

### Run Tests with `pytest`

Ensure that `pytest` is installed in your conda environment. To run the tests, simply use:

```bash
pytest
```

### Check Syntax with `pre-commit`

The PanSeg repository uses pre-commit hooks to ensure the code is correctly formatted and free of linting issues. While not mandatory, it is encouraged to check your code before committing by running:

```bash
pre-commit run --all-files
```

Commit messages are important. Please follow the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) specification.

For efficiency, pytest is not included in the pre-commit hooks. Please run the tests separately.

## Release

The versions of most dependencies in the conda package are pinned.
Before releasing, you might want to update those pins:

```bash
bash update_deps.sh
```

This updates the environment `panseg-dev` and its `yaml` file.  
It also updates the `environment.yaml` by matching its content against the now
updated packages in `environment-dev.yaml`.

Using this updated `environment.yaml` file, **manually** update the following files:

* `conda-recipe/meta.yaml`

To release, run the github workflow `release`, then manually publish to `conda-forge`.

### Github release workflow

The workflow will:

* run tests
* bump version number
* create a github release
* build a conda package
* upload to github
* build documentation
* build an installer for win/mac/linux
* upload installer to heibox

### Conda-Forge release

**Run after gh release workflow**

* git clone personal panseg-feedstock fork
* update `recipe/meta.yaml` version pins!
* Update hash and version in `recipe/meta.yaml`
  * `curl -sL https://github.com/kreshuklab/panseg/archive/refs/tags/VERSION.tar.gz | openssl sha256`
* Create PR
* Re-render
* Merge

## Videos

Video encoding settings for this website:

```bash
ffmpeg -i input.webm -vcodec libx264 -r 20 -crf 28 -tune animation -vf "scale=-2:'min(1080,ih)'" output_20fps.mp4
```
