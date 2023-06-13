# Stochastic-Deep Learning Parameterization of Ocean Momentum Forcing

This repository provides a subgrid model of ocean momentum forcing, based on a
convolutional neural network (CNN) trained on high-resolution
surface velocity data from CM2.6.
This model can then be coupled into larger GCMs, e.g., at coarser granularity to provide
high-fidelity parameterization of ocean momentum forcing.
The parameterization output by the CNN consists of a Gaussian distribution
specified by 2 parameters (mean and standard deviation), which allows for
stochastic implementations in online models.

The model is based on the paper [Stochastic-Deep Learning Parameterization of Ocean Momentum Forcing,
Arthur P. Guillaumin, Laure Zanna](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2021MS002534).
The exact version of the code used to produce said paper can be found on [Zenodo](https://zenodo.org/record/5076046#.ZF4ulezMLy8).
The present repository provides a version of this model which is designed for others to reproduce, replicate,
and reuse.

__This repository is currently work-in-progress following a process of refreshing
the code and making it available for easy reuse by others__

# Using this software
The model is packaged with `pyproject.toml`/setuptools. For local development and
testing, run `pip install -e .` in the root directory.

An alternate `pyproject.toml` file is provided for building with Poetry. To use,
rename `pyproject-poetry.toml` to `pyproject.toml` (clobbering the existing
file) and use Poetry as normal.

## License
Provided under the MIT license. See `LICENSE` for license text.

## Contributing
We are not currently accepting contributions outside of the M2LInES and ICCS projects until we have
reached a code release milestone.

# General code documentation

In order to keep track of script runs, we have used MLFLOW (https://mlflow.org/) to organize our code. In particular we define three important types
of experiments, each corresponding to a different script:

- data type: cmip26.py
- training type: trainScript.py
- testing type: testing/main.py

The data type experiment corresponds to different runs of the scirpt cmip26.py generating the coarse surface velocities, as well as the diagnosed forcing.

The training type experiment corresponds to different runs of trainScript.py, resulting in a trained model. Various runs might used different neural networks, loss
functions, number of training epochs, or training data. For the latter, it makes sense that one will need to specify a run from the data type experiment
where the training data will be collected for training. Different runs are also used for different values of CO2.

The testing type uses a model from the training type experiment and applies it, without further training, to global data. The global data used for the test
is, again, specified as a run from the data type experiment.

## Specific to Greene implementation

To list the currently existing experiments, one can follow this procedure:

- Start a singularity container with the appropriate image via the following command:
```
singularity exec --nv --overlay /scratch/ag7531/overlay2 /scratch/work/public/singularity/cuda11.0-cudnn8-devel-ubuntu18.04.sif /bin/bash
```
- Activate conda:
```
source /ext3/env.sh
```
- Activate the subgrid conda environment:
```
conda activate subgrid
```

Finally, start python, import the mlflow module and call the mlflow method that allows to list expriments:
```
from mlflow.tracking import client
cl = client.MlflowClient()
for e in cl.list_experiments():
    print(e.experiment_id, e.name)
```

If nothing shows, return to bash, create the following environment variable to tell mlflow where to look for experiments,
```
export MLFLOW_TRACKING_URI=/scratch/ag7531/mlruns
```

and run the python lines again.

Right now, there are many experiments, including some old ones which are no more used. But the most important ones are:
- for data: data-global, with id 19
- for models: modelsv1, with id 21
- for testing: test_global, with id 20

# Run data code


Simple way: modify $cmd in /home/ag7531/jobs/job-forcingdata.sh with the parameters described further down (e.g. loss_cls_name etc) and run
```
sbatch /home/ag7531/jobs/job-forcingdata.sh
```

Generating coarse velocity data and diagnosed forcing is achieved by running the following command:

```
mlflow run git@github.com:arthurBarthe/subgrid.git --experiment-name data-global --env-manager=local -P lat_min=-85 -P lat_max=85 -P long_min=-280 -P long_max=80 -P factor=4 -P chunk_size=1 -P CO2=1 -P global=1
```

For older MLflow versions, replace `--env-manager=local` with `--no-conda`.

The CLI parameters one may want to change are:
- experiment-name: the name of the data experiment under which the run will be saved. This will be used later on to recover the generated data for
either training or testing.
- factor: the factor definining the low-resolution grid of the generated data with respect to the high-resolution grid.
- CO2: 0 for control, 1 for 1% increase per year dataset.

The rest of the CLI parameters can be kept as such.

# Run training code

Simple way: modify $cmd in /home/ag7531/jobs/job-training3.2.sh with the parameters described further down (e.g. loss_cls_name etc) and run
```
sbatch /home/ag7531/jobs/job-training3.2.sh
```

Training is achieved by running trainScript.py. This scripts accepts a range of CLI parameters. The simplest way to run the code is via the following MLFLOW command,
which runs this script as an MLFLOW project.

```
mlflow run git@github.com:Zanna-ResearchTeam/subgrid.git --experiment-name new_models -e train --no-conda -P exp_id=19 -P run_id=afab43b4a6274e29822181c4fdaaf925 -P learning_rate=0/5e-4/15/5e-5/30/5e-6 -P n_epochs=200 -P weight_decay=0.00 -P train_split=0.8 \
-P test_split=0.85 -P model_module_name=models.models1 -P model_cls_name=FullyCNN -P batchsize=4 -P transformation_cls_name=SoftPlusTransform -P submodel=transform3 -P loss_cls_name=HeteroskedasticGaussianLossV2
```

This uses the last version of the code on github rather than the local version. Most of the CLI parameters can be kept as such. Some one might want to change are:
- experiment-name: name of the experiment under which the run will be recorded. In particular, this will be used to recover the trained neural network.
- exp_id: id of the experiment containing the run that generated the forcing data.
- run_id: id of the run that generated the forcing data that will be used for training.
- loss_cls_name: name of the class that defines the loss. This class should be defined in train/losses.py in order for the script to find it. Currently the
main available options are:
    - HeteroskedasticGaussianLossV2: this corresponds to the loss used in the paper
    - BimodalGaussianLoss: a Gaussian loss defined using two Gaussian modes
- model_module_name: name of the module that contains the class defining the NN used
- model_cls_name: name of the class defining the NN used, should be defined in the module specified by model_module_name


Another important way to modify the way the script runs consists in modifying the domains used for training. These are defined in training_subdomain.yaml in terms
of their coordinates. Note that at run time domains will be truncated to the size of the smallest domain in terms of number of points.

# Run Testing code
Testing is carried out in an interactive sbatch run, where we ask for a GPU:

```
srun -t1:30:00 --mem=40000 --gres=gpu:1 --pty /bin/bash
```

We then start our singularity container and activate the subgrid conda environment:

- Start a singularity container with the appropriate image via the following command:
```
singularity exec --nv --overlay /scratch/ag7531/overlay2 /scratch/work/public/singularity/cuda11.0-cudnn8-devel-ubuntu18.04.sif /bin/bash
```
- Activate conda:
```
source /ext3/env.sh
```
- Activate the subgrid conda environment:
```
conda activate subgrid
```

We then cd to /home/ag7531/code/subgrid and run:

```
python -m testing.main.py --n_splits 50 --batch_size 2 --n_test_times 7300
```




ADD the list of possible options for the loss / neural network model
Create separate mlflow experiments for Laure
Clean up experiments
Give read x access to Laure
Make sure there is backup for stuff on scratch
Update analysis code so that it produces the nice figures in the paper


Add something about how to load the NN independently.
