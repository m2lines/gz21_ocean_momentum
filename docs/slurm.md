# Old Slurm notes
The repository's original readme included lots of notes on running this software
on Slurm, but referenced missing files. These notes are migrated here.

---

## Specific to Greene implementation
To list the currently existing experiments, one can follow this procedure:

1. Start a singularity container with the appropriate image via the following
   command:

```
singularity exec --nv --overlay /scratch/ag7531/overlay2 /scratch/work/public/singularity/cuda11.0-cudnn8-devel-ubuntu18.04.sif /bin/bash
```

2. Activate conda:

```
source /ext3/env.sh
```

3. Activate the subgrid conda environment:

```
conda activate subgrid
```

4. Start python, import the MLflow module and list experiments:

```python
from mlflow.tracking import client
cl = client.MlflowClient()
for e in cl.list_experiments():
    print(e.experiment_id, e.name)
```

If nothing is displayed, return to Bash and create the following environment
variable to tell MLflow where to look for experiments:

    export MLFLOW_TRACKING_URI=/scratch/ag7531/mlruns

and run the above Python block again.

Right now, there are many experiments, including some old ones which are no more used. But the most important ones are:
- for data: data-global, with id 19
- for models: modelsv1, with id 21
- for testing: test_global, with id 20

# Run data code
Simple way: modify $cmd in /home/ag7531/jobs/job-forcingdata.sh with the
parameters described further down (e.g. loss_cls_name etc) and run

    sbatch /home/ag7531/jobs/job-forcingdata.sh

Generating coarse velocity data and diagnosed forcing is achieved by running the
following command:

```
mlflow run git@github.com:arthurBarthe/subgrid.git --experiment-name data-global --env-manager=local -P lat_min=-85 -P lat_max=85 -P long_min=-280 -P long_max=80 -P factor=4 -P chunk_size=1 -P CO2=1 -P global=1
```

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
