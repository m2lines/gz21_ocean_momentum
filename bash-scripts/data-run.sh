#!/bin/bash
#$ -cwd     
#$ -pe smp 1      # Request 1 core
#$ -l h_rt=1:0:0  # Request 1 hour runtime
#$ -l h_vmem=10G   # Request 1GB RAM

cd ..
source ~/.bashrc
poetry run mlflow run --experiment-name data-global --env-manager local -P lat_min=-85 -P lat_max=85 -P long_min=-280 -P long_max=80 -P factor=4 -P chunk_size=1 -P CO2=1 -P global=0 -P ntimes=100 .
