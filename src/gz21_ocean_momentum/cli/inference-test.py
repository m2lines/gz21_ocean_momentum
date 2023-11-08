import configargparse

import logging

from gz21_ocean_momentum.train.base import Trainer

# TODO hardcode submodel, transformation, NN loss function
# unlikely for a CLI we need to provide dynamic code loading -- let's just give
# options
# we could enable such "dynamic loading" in the "library" interface!-- but, due
# to the class-based setup, it's a little complicated for a user to come in with
# their own code for some of these, and it needs documentation. so a task for
# later
import gz21_ocean_momentum.models.models1.FullyCNN as model_cls
import gz21_ocean_momentum.models.submodels.transform3 as submodel
import gz21_ocean_momentum.models.transforms.SoftPlusTransform as transformation
import gz21_ocean_momentum.train.losses.HeteroskedasticGaussianLossV2 as loss_cls

from gz21_ocean_momentum.data.datasets import
    pytorch_dataset_from_cm2_6_forcing_dataset

DESCRIPTION = """
Use pre-trained GZ21 neural net to predict forcing for input ocean velocity data.

This script is intended as example of how use the GZ21 neural net, and for
general tinkering.

Designed to ingest coarsened CM2.6 data: looks for data variables at certain
names (`xu_ocean`, ...) with certain units. If these do not match up, the neural
net will not operate properly.

More specifically, this script is designed to ingest coarsened CM2.6 data as
output from the GZ21 data step. This also computes forcings, which are ignored.
(Ideally, we would provide a short script to simply coarsen some data.)

Note that the neural net has two outputs per grid point. See project
documentation (specifically `README.md` in the project repository), and the
associated paper Guillaumin (2021) for suggestions on how to integrate these
into your GCM of choice.
"""

p = configargparse.ArgParser(description=DESCRIPTION)
p.add("--config-file", is_config_file=True, help="config file path")

p.add("--input-data-dir", type=str, required=True, help="path to input ocean velocity data, in zarr format (folder)")
p.add("--model-state-dict-file", type=str, required=True, help="model state dict file (*.pth)")
p.add("--device",  type=str, default="cuda", help="neural net device (e.g. cuda, cuda:0, cpu)")
p.add("--out-dir", type=str, required=True,  help="folder to save forcing predictions dataset to (in zarr format)")

p.add("--verbose", action="store_true", help="be more verbose (displays progress, debug messages)")

options = p.parse_args()

# set up logging immediately after parsing CLI options (need to check verbosity)
# (would like to simplify this, maybe with `basicConfig(force=True)`)
if options.verbose:
    logging.basicConfig(level=logging.DEBUG)
    dask.diagnostics.ProgressBar().register()
    logger = logging.getLogger(__name__)
    logger.debug("verbose mode; displaying all debug messages, progress bars)")
else:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

logger.debug("dropping irrelevant data variables...")
surface_fields = surface_fields[["usurf", "vsurf"]]

if options.ntimes is not None:
    logger.info(f"slicing {options.ntimes} time points...")
    surface_fields = surface_fields.isel(time=slice(options.ntimes))

logger.info("selecting input data bounding box...")
surface_fields = bounding_box.bound_dataset("yu_ocean", "xu_ocean", surface_fields, bbox)

# ---

# TODO hard-coded loss class
criterion = loss_cls(dataset.n_targets)

# load, prepare pre-trained neural net
net = model_cls(dataset.n_features, criterion.n_required_channels)
net.load_state_dict(torch.load(options.model_state_dict_file))
print(net)
#net.cpu() # TODO why needed?
dataset.add_transforms_from_model(net)

print("Size of training data: {}".format(len(train_dataset)))
print("Size of validation data : {}".format(len(test_dataset)))
print("Input height: {}".format(train_dataset.height))
print("Input width: {}".format(train_dataset.width))
print(train_dataset[0][0].shape)
print(train_dataset[0][1].shape)
print("Features transform: ", transform.transforms["features"].transforms)
print("Targets transform: ", transform.transforms["targets"].transforms)

# Net to GPU
with TaskInfo("Put neural network on device"):
    net.to(options.device)

print("width: {}, height: {}".format(dataset.width, dataset.height))

with ProgressBar(), TaskInfo("Predict & save prediction dataset"):
    out = predict_lazy_cm2_6(net, criterion, partition, loaders, options.device)
    ProgressBar().register()
    logger.info(f"chunk predictions to time=32 ...")
    out = out.chunk(dict(time=32))
    print(f"Size of output data is {out.nbytes/1e9} GB")
    logger.info(f"writing re-chunked predictions zarr to directory: {options.out_dir}")
    out.to_zarr(options.out_dir)
