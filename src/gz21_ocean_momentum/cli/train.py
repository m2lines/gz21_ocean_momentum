import gz21_ocean_momentum.step.train.lib as lib

import configargparse

DESCRIPTION = "GZ21 train step: train a Pytorch model on input forcing data."

p = configargparse.ArgParser()
p.add("--config-file", is_config_file=True, help="config file path")
p.add("--batchsize", type=int, required=True)
p.add("--epochs", type=int, required=True)
p.add("--out-model", type=str, required=True, help="export trained model to this path")

options = p.parse_args()

model = lib.do_something()
torch.save(model.state_dict(), options.out_model)
