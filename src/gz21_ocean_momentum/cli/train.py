import gz21_ocean_momentum.step.train.lib as lib
import gz21_ocean_momentum.common.cli as cli
import gz21_ocean_momentum.common.assorted as common
from   gz21_ocean_momentum.common.bounding_box import load_bounding_boxes_yaml

import configargparse

DESCRIPTION = "GZ21 train step: train a Pytorch model on input forcing data."

p = configargparse.ArgParser()
p.add("--config-file", is_config_file=True, help="config file path")
p.add("--forcing-data-dir", type=str, required=True, help="directory containing zarr-format forcing data to use")
p.add("--subdomains-file", type=str, required=True, help="YAML file describing subdomains to use (bounding boxes). TODO format")
p.add("--batchsize", type=int, required=True)
p.add("--epochs", type=int, required=True)
p.add("--out-model", type=str, required=True, help="export trained model to this path")
p.add("--initial-learning-rate", type=float, required=True, help="initial learning rate for optimization algorithm")
p.add("--epoch-milestones", type=float, action="append", required=True, help="TODO. specify multiple times to form a list. must be strictly increasing, no dupes")
p.add("--decay-factor", type=float, required=True, help="learning rate decay factor, applied each time an epoch milestone is reached")

options = p.parse_args()

if not common.list_is_strictly_increasing(options.epoch_milestones):
    cli.fail(2, "epoch milestones list is not strictly increasing")

sds = lib.select_subdomains(
        xr.open_zarr(options.forcing_data_dir),
        load_bounding_boxes_yaml(options.subdomains_file)

#net = lib.do_something()
#torch.save(model.state_dict(), options.out_model)
