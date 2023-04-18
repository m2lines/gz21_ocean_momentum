#!/bin/bash
#$ -cwd     
#$ -pe smp 1      # Request 1 core
#$ -l h_rt=1:0:0  # Request 1 hour runtime
#$ -l h_vmem=10G   # Request 1GB RAM

cd ..
source ~/.bashrc
# in the command below the parameter run_id is the id of the run that generated the dataset that should be used for
# training.
poetry run mlflow run --experiment-name new_models -e train --env-manager local -P \
run_id=53e96b1343214a37bc18f3754d476323 -P learning_rate=0/5e-4/15/5e-5/30/5e-6 \
-P n_epochs=200 -P weight_decay=0.00 -P train_split=0.8 \
-P test_split=0.85 -P model_module_name=models.models1 -P model_cls_name=FullyCNN\
 -P batchsize=4 -P transformation_cls_name=SoftPlusTransform -P submodel=transform3 -P loss_cls_name=HeteroskedasticGaussianLossV2 .
