# Stochastic-Deep Learning Parameterization of Ocean Momentum Forcing
[gz21-paper-code-zenodo]: https://zenodo.org/record/5076046#.ZF4ulezMLy8
[gz21-paper-agupubs]: https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2021MS002534

This repository provides a subgrid model of ocean momentum forcing, based on a
convolutional neural network (CNN) trained on high-resolution surface velocity
data from CM2.6. This model can then be coupled into larger GCMs, e.g., at
coarser granularity to provide high-fidelity parameterization of ocean momentum
forcing. The parameterization output by the CNN consists of a Gaussian
distribution specified by 2 parameters (mean and standard deviation), which
allows for stochastic implementations in online models.

The model is based on the paper [Stochastic-Deep Learning Parameterization of
Ocean Momentum Forcing, Arthur P. Guillaumin, Laure
Zanna][gz21-paper-agupubs]. The exact version of the code used to produce said
paper can be found on [Zenodo][gz21-paper-code-zenodo]. The present repository
provides a version of this model which is designed for others to reproduce,
replicate, and reuse.

__This repository is currently work-in-progress following a process of refreshing
the code and making it available for easy reuse by others.__

## Architecture
The model is written in Python, using PyTorch for the CNN. We provide 3 separate
"stages", which are run using different commands and arguments:

  * data processing: downloads part of CM2.6 dataset and processes
  * model training: train model on processed data
  * model testing: tests the trained model on an unseen region

For more details, see the `docs` directory.

## Usage
### Dependencies
Python 3 is required.

#### Python
With `pip` installed, run the following in the root directory:

    pip install -e

*(An alternate `pyproject.toml` file is provided for building with Poetry. To
use, rename `pyproject-poetry.toml` to `pyproject.toml` (clobbering the existing
file) and use Poetry as normal. Note that the Poetry build is not actively
supported-- if it fails, check that the dependencies are up to date with the
setuptools `pyproject.toml`.)*

#### System
Some graphing code uses cartopy, which requires [GEOS](https://libgeos.org/). To
install on Ubuntu:

    sudo apt install libgeos-dev

On MacOS, via Homebrew:

    brew install geos

On Windows, consider using MSYS2 to install the library in a Linux-esque manner:
https://packages.msys2.org/package/mingw-w64-x86_64-geos

## Contributing
We are not currently accepting contributions outside of the M2LInES and ICCS projects until we have
reached a code release milestone.

## License
This repository is provided under the MIT license. See `LICENSE` for license
text and copyright information.

---

For older MLflow versions, replace `--env-manager=local` with `--no-conda`.

The CLI parameters one may want to change are:
- experiment-name: the name of the data experiment under which the run will be saved. This will be used later on to recover the generated data for
either training or testing.
- factor: the factor definining the low-resolution grid of the generated data with respect to the high-resolution grid.
- CO2: 0 for control, 1 for 1% increase per year dataset.

Most of the CLI parameters can be kept as such. Some one might want to change are:

- `experiment-name`: name of the experiment under which the run will be
  recorded. In particular, this will be used to recover the trained neural
  network.
- `exp_id`: id of the experiment containing the run that generated the forcing
  data.
- `run_id`: id of the run that generated the forcing data that will be used for
  training.
- `loss_cls_name`: name of the class that defines the loss. This class should be
  defined in train/losses.py in order for the script to find it. Currently the
  main available options are:
  - `HeteroskedasticGaussianLossV2`: this corresponds to the loss used in the
    paper
  - `BimodalGaussianLoss`: a Gaussian loss defined using two Gaussian modes
- `model_module_name`: name of the module that contains the class defining the
  NN used
- `model_cls_name`: name of the class defining the NN used, should be defined in
  the module specified by model_module_name

Another important way to modify the way the script runs consists in modifying
the domains used for training. These are defined in `training_subdomain.yaml` in
terms of their coordinates. Note that at run time domains will be truncated to
the size of the smallest domain in terms of number of points.
