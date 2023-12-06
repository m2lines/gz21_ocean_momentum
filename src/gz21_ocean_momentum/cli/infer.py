import configargparse

import gz21_ocean_momentum.common.cli as cli
import logging
from dask.diagnostics import ProgressBar
from gz21_ocean_momentum.utils import TaskInfo

import gz21_ocean_momentum.lib.model as lib
from gz21_ocean_momentum.data.datasets import (
    #DatasetPartitioner,
    DatasetTransformer,
    DatasetWithTransform,
    ComposeTransforms,
)

import xarray as xr
import torch
from torch.utils.data import DataLoader

import gz21_ocean_momentum.models.models1 as model
import gz21_ocean_momentum.models.submodels as submodels
import gz21_ocean_momentum.models.transforms as transforms
import gz21_ocean_momentum.train.losses as loss_funcs
from gz21_ocean_momentum.inference.utils import predict_lazy_cm2_6

# Description of this module
_cli_desc = """
Use a trained GZ21 neural net to predict forcing for input ocean velocity data.

This script is intended as example of how use the GZ21 neural net, generating
data for analyzing and visualizing model behaviour, and for general tinkering.

Designed to ingest coarsened CM2.6 data: looks for data variables at certain
names (`xu_ocean`, ...) with certain units. If these do not match up, the neural
net will not operate properly.

More specifically, this script is designed to ingest coarsened CM2.6 data as
output from the GZ21 data step. This also computes forcings, which are ignored.
(Ideally, we would provide a short script to simply coarsen some data, without
computing the associated forcings.)

Note that the neural net has two outputs per grid point. See project
documentation (specifically `README.md` in the project repository), and the
associated paper Guillaumin (2021) for suggestions on how to integrate these
into your GCM of choice.
"""

submodel = submodels.transform3

p = configargparse.ArgParser(description=_cli_desc)
p.add("--config-file", is_config_file=True, help="config file path")
p.add("--input-data-dir", type=str, required=True, help="path to input ocean velocity data, in zarr format (folder)")
p.add("--model-state-dict-file", type=str, required=True, help="model state dict file (*.pth)")
p.add("--out-dir", type=str, required=True,  help="folder to save forcing predictions dataset to (in zarr format)")
p.add("--device",  type=str, default="cuda", help="neural net device (e.g. cuda, cuda:0, cpu)")

options = p.parse_args()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

cli.fail_if_path_is_nonempty_dir(
        1, f"--out-dir \"{options.out_dir}\" invalid", options.out_dir)

# ---

logger.info("loading input (coarse) ocean momentum data...")
ds_computed_xr = xr.open_zarr(options.input_data_dir)

with ProgressBar(), TaskInfo("Applying transforms to dataset"):
    ds_computed_xr = submodel.fit_transform(ds_computed_xr)

# wrap xarray into PyTorch-compatible data
dataset = lib.gz21_train_data_subdomain_xr_to_torch(ds_computed_xr)
loader = DataLoader(dataset)

criterion = loss_funcs.HeteroskedasticGaussianLossV2(dataset.n_targets)
net = model.FullyCNN(dataset.n_features, criterion.n_required_channels)

# load final net transformation
# (this is correct, assuming any transformation state if present is stored in
# the model state dict)
transformation = transforms.SoftPlusTransform()
transformation.indices = criterion.precision_indices
net.final_transformation = transformation

net.load_state_dict(torch.load(options.model_state_dict_file))

dataset.add_transforms_from_model(net)

with TaskInfo(f"moving neural network to requested device: {options.device}"):
    net.to(options.device)

with ProgressBar(), TaskInfo("Predict & save prediction dataset"):
    out = predict_lazy_cm2_6(net,
                             criterion.n_required_channels,
                             criterion.channel_names,
                             [dataset], [loader], options.device)
    ProgressBar().register()
    logger.info(f"chunk predictions to time=32 ...")
    out = out.chunk(dict(time=32))
    print(f"Size of output data is {out.nbytes/1e9} GB")
    logger.info(f"writing re-chunked predictions zarr to directory: {options.out_dir}")
    out.to_zarr(options.out_dir)
