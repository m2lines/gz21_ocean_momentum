# Reproducing results from the Guillaumin (2021) paper
See README.md at repo root for further details. Here, we will provide just
commands and commentary.

Options such as `--out-dir` will be omitted. (The script will prompt you for
missing options. You can display help by adding `--help` to your invocation.)

## 1. Training data generation
```
python src/gz21_ocean_momentum/cli/data.py \
--config-file resources/cli-configs/data-paper.yaml
```

Unclear whether you may need `--ntimes 4000`.

## 2. Model training
Not tested due to issues with training.

Model hyperparameters adapted from Table A1.

```
python src/gz21_ocean_momentum/cli/train.py \
--config-file resources/cli-configs/train-paper.yaml \
--subdomains-file resources/cli-configs/train-subdomains-paper.yaml \
--train-split-end 0.8 --test-split-start 0.85
```

Add `--in-train-data-dir <forcings generated above>`.

## 3. Inference
The CLI inference script has no configuration other than model to predict on,
and input low-resolution data to predict forcings of:

    python src/gz21_ocean_momentum/cli/infer.py

Currently will not reproduce the same predictions as used in the paper. See
https://github.com/m2lines/gz21_ocean_momentum/pull/97 for further details.

For `--model-state-dict-file`, you may use a pretrained model instead of running
the training described above. A low-resolution one is provided here:
https://huggingface.co/M2LInES/gz21-ocean-momentum/blob/main/low-resolution/files/trained_model.pth

Similarly, instead of generating forcings as above, you may use pre-generated
data for `--input-data-dir`. Low-resolution (~100 timepoints) CM2.6 data:
https://huggingface.co/datasets/M2LInES/gz21-forcing-cm26/tree/main/forcing
