# Reproducing results from the Guillaumin (2021) paper
See README.md at repo root for further details. Here, we will provide just
commands and commentary.

## 1. Training data generation
```
python src/gz21_ocean_momentum/cli/data.py \
--config-file examples/cli-configs/data-paper.yaml \
--out-dir $(mktemp -d)
```

Unclear whether you may need `--ntimes 4000`.

Configure `--out-dir` as you like.

## 2. Model training
WIP.

```
python src/gz21_ocean_momentum/cli/train.py
--subdomains-file examples/cli-configs/train-subdomains-paper.yaml
--batch-size 8 --epochs 20
--initial-learning-rate 5.0e-4 --decay-factor 0.0 --decay-at-epoch-milestones 15
--decay-at-epoch-milestones 30 --train-split 0.8 --test-split 0.85 \
--device cuda:0 \
--in-train-data-dir tmp/generated/forcings/paper-fig-1-ctrl-n100
--out-model $(mktemp -d)
```

Configure `--out-model` as you like.

## 3. Inference
The CLI inference script has no configuration other than model to predict on,
and input low-resolution data to predict forcings of:

```
python src/gz21_ocean_momentum/cli/infer.py \
--input-data-dir tmp/forcings/example \
--model-state-dict-file tmp/models/example.pth \
--out-dir $(mktemp -d)
```

You may use a pretrained model. A low-resolution one is provided here:
https://huggingface.co/M2LInES/gz21-ocean-momentum/blob/main/low-resolution/files/trained_model.pth

Example low resolution CM2.6 data for predicting on is available here:
https://huggingface.co/datasets/M2LInES/gz21-forcing-cm26/tree/main/forcing
