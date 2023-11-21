# GZ21: stochastic deep learning parameterization of ocean momentum forcing
[gz21-paper-code-zenodo]: https://zenodo.org/record/5076046#.ZF4ulezMLy8
[gz21-paper-agupubs]: https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2021MS002534
[cm26-ds]: https://catalog.pangeo.io/browse/master/ocean/GFDL_CM2_6/

This repository trains a PyTorch convolutional neural network (CNN) to predict
subgrid ocean momentum forcing from ocean surface velocity, intended for
coupling with larger GCMs to provide a performant, high-fidelity
parameterization in coarse-resolution climate models.

Scripts for preparing training data, training up a model, and using the model to
make predictions (inference mode) are provided. These are run from the
command-line, and accept various configuration options (e.g. hyperparameters for
NN training).

For further detail and discussion, please read the original paper
[Arthur P. Guillaumin, Laure Zanna (2021). Stochastic-deep learning
parameterization of ocean momentum forcing][gz21-paper-agupubs].
Documentation in this repository will refer back to sections from the paper e.g.
*Guillaumin (2021) 2.1* to provide context and further reading.
(A snapshot of the code used in the paper can be found on
[Zenodo][gz21-paper-code-zenodo].)

## Overview
Most of this repository is concerned with preparing training data, and training
a NN. Each of these is handled with a standalone command-line interface (CLI)
Python script, and data is saved and loaded between via disk.

In the "data" step, we generate training data using
[simulation data from the CM2.6 climate model][cm26-ds]
(which we refer to as the CM2.6 dataset, or just CM2.6).
We calculate the subgrid forcing needed for coarse-resolution models using the
high-resolution ocean velocity data in the CM2.6 dataset, then coarsen. This
coarsened, with-forcings dataset is saved to disk. You may generate training
data using either the "control" CM2.6 simulation, or the "1-percent annual CO2
increase" one. *(See Guillaumin (2021) 2.1.)*

In the "training" step, we train a NN to predict the true forcing from the
coarse velocity data generated above. This forcing term tends to have a large
amount of uncertainty. Rather than a single value, we predict both the mean and
standard deviation of a Gaussian probability distribution for the forcing. This
allows for stochastic implementations in online models. *(See Guillaumin (2021)
2.3 for a more in-depth explanation and how to interpret the NN output.)*

In the "testing" step, we test a trained model on an unseen region of data (the
subset not used in the previous training step).

### Repository layout
* `src`: source code (both library functions and CLI scripts)
* `tests`: pytest tests
* `docs`: detailed project documentation, implementation notes
* `examples`: CLI step configs, Jupyter notebooks for generating figures etc.
* `flake.nix`, `flake.lock`: helper files for building on Nix (ignore)

## Installation
Python 3.9 or newer is required. We primarily test on Python 3.11.

To avoid any conflicts with local packages, we recommend using a virtual
environment. In the root directory:

    python -m venv venv

or using [virtualenv](https://virtualenv.pypa.io/en/latest/):

    virtualenv venv

Then load with `source venv/bin/activate`.

With `pip` installed, run the following in the root directory:

    pip install -e .

*(An alternate `pyproject.toml` file is provided for building with
[Poetry](https://python-poetry.org/). To use, rename `pyproject-poetry.toml` to
`pyproject.toml` (overwriting the existing file) and use Poetry as normal. Note
that the Poetry build is not actively supported-- if it fails, check that the
dependencies are up-to-date with the setuptools `pyproject.toml`.)*

Note that if you are running Python 3.9 or older, you may also need to install
the [GEOS](https://libgeos.org/) library, due to `cartopy` requiring it. (Newer
versions moved away from the C dependency.)

## Usage
Execute these commands from the repository root.

See [`docs`](docs/) directory for more details.

For command-line option explanation, run the appropriate step with `--help` e.g.
`python src/gz21_ocean_momentum/cli/data.py --help`.

Most CLI scripts support reading in options from a YAML file using a
`--config-file` flag. In general, a flag `--name value` will be converted to a
top-level `name: value` line. Examples are provided in
[`examples/cli-configs`](examples/cli-configs/). CLI options override file
options, so you may provide partial configuration in a file and fill out the
rest (e.g. file paths) on the command line.

### Unit tests
There are a handful of unit tests using pytest, in the [`tests`](tests/)
directory. These assert some operations and methods used in the steps. They may
be run in the regular method:

    pytest

### Training data generation
[`cli/data.py`](src/gz21_ocean_momentum/cli/data.py) calculates coarse
surface velocities and diagnosed forcings from the CM2.6 dataset and saves them
to disk. This is used as training data for the neural net.

**You must configure GCP credentials in order to download the CM2.6 dataset.**
See [`docs/data.md`](docs/data.md) for more details.

Example invocation:

    python src/gz21_ocean_momentum/cli/data.py \
    --lat-min -80 --lat-max 80 --long-min -280 --long-max 80 \
    --factor 4 --ntimes 100 --co2-increase --out-dir forcings

Alternatively, you may write (all or part of) these options into a YAML file:

```yaml
lat-min:  -80
lat-max:  80
long-min: -280
long-max: 80
ntimes: 100
factor: 4
co2-increase: true
```

and use this file in an invocation with the `--config-file` option:

    python src/gz21_ocean_momentum/cli/data.py \
    --config-file examples/cli-configs/data-paper.yaml --out-dir forcings

Some preprocessed data is hosted on HuggingFace at
[datasets/M2LInES/gz21-forcing-cm26](https://huggingface.co/datasets/M2LInES/gz21-forcing-cm26).

You may also run the data processing step directly from Python using the
functions at [`step/data/lib.py`](src/gz21_ocean_momentum/step/data/lib.py). See
the CLI script for example usage.

#### Training
[cli-train]: src/gz21_ocean_momentum/cli/train.py

The [`cli/train.py`][cli-train] script runs the model training step. You may
configure various training parameters through command-line arguments, such as
number of training epochs, loss functions, and training data. (You will want to
select the output from a data processing step for the latter.)

Example invocation:

```
python src/gz21_ocean_momentum/cli/train.py \
--lat-min -80 --lat-max 80 --long-min -280 --long-max 80 \
--factor 4 --ntimes 100 --co2-increase --out-dir forcings \
--train-split-end 0.8 --test-split-start 0.85 \
--subdomains-file examples/cli-configs/training-subdomains-paper.yaml \
--forcing-data-path <forcing zarr dir>
```

You may place options into a YAML file and load with the `--config-file` option.


Plain Python call example:

Relevant parameters:

* `--subdomains-file`: path to YAML file storing a list of subdomains to select
  from the forcing data, which are then used for training. (Note that at
  runtime, domains are be truncated to the size of the smallest domain in terms
  of number of points.)
* `--train-split-end`: use `0->N` percent of the dataset for training
* `--test-split-start`: use `N->100` percent of the dataset for testing

##### Subdomains
The `--subdomains-file` format is a YAML list of bounding boxes, each defined
using four labelled floats:

```yaml
- lat-min: 35
  lat-max: 50
  long-min: -50
  long-max: -20
- lat-min: -40
  lat-max: -25
  long-min: -180
  long-max: -162
# - ...
```

`lat-min` must be smaller than `lat-max`, likewise for `long-min`.

*Note:* Ensure that the subdomains you use are contained in the domain of the
forcing data you use. If they aren't, you may get a confusing Python error along
the lines of:

```
RuntimeError: Calculated padded input size per channel: <smaller than 5 x 5>.
Kernel size: (5 x 5). Kernel size can't be greater than actual input size
```

#### Inference
The [`inference/main.py`](src/gz21_ocean_momentum/inference/main.py) script runs the
model testing stage. This consists of running a trained model on a dataset. 
The model's output are then stored as an artefact. This step
should ideally be run with a GPU device available, to achieve a better speed.

In this step it is particularly important to set the environment variable `MLFLOW_TRACKING_URI`
in order for the data to be found and stored in a sensible place.

One can run the inference step by interactively
running the following in the project root directory:

    python3 -m gz21_ocean_momentum.inference.main --n_splits=40

with `n_splits` being the number of subsets which the dataset is split 
into for the processing, before being put back together for the final output.
This is done in order to avoid memory issues for large datasets.
Other useful arguments for this call would be 
- `to_experiment`: the name of the mlflow experiment used for this run (default is "test").
- `batch_size`: the batch size used in running the neural network on the data.


After the script has started running, it will first require
the user to select an experiment and a run corresponding to a 
training step run previously. 
The user will then be required to select an experiment and a run
corresponding to a data step previously run.

The inference step should then start.

### Jupyter Notebooks
The [examples/jupyter-notebooks](examples/jupyter-notebooks/) folder stores
notebooks developed during early project development, some of which were used to
generate figures used in the 2021 paper. See the readme in the above folder for
details.

### Dev Branch
The `dev` branch contains ongoing refactoring work which removes the necessity to use
mlflow. Currently, the code has been refactored into a clearer structure and easier
use through a command line interface for the data step, and the training step
is in progress. Further work is needed for the inference step, and to adapt the Jupyter
notebooks.

## Data on HuggingFace
There is GZ21 Ocean Momentum data available on [HuggingFace](https://huggingface.co/)

* [the output of the data step][datasets/M2LInES/gz21-forcing-cm26] and
* [the trained model](https://huggingface.co/M2LInES/gz21-ocean-momentum).

## Contributing
We are not currently accepting contributions outside of the M2LInES and ICCS
projects until we have reached a code release milestone.

## License
This repository is provided under the MIT license. See [`LICENSE`](LICENSE) for
license text and copyright information.

## Citing this software
See `CITATION.cff` for citation metadata for this software. *(For further
details on usage, see https://citation-file-format.github.io/ .)*
