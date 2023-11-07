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

DESCRIPTION = "GZ21 inference step: predict forcings trained model on "

p = configargparse.ArgParser(description=DESCRIPTION)
p.add("--config-file", is_config_file=True, help="config file path")

p.add("--lat-min",  type=float, required=True, help="bounding box minimum latitude")
p.add("--lat-max",  type=float, required=True, help="bounding box maximum latitude")
p.add("--long-min", type=float, required=True, help="bounding box minimum longitude")
p.add("--long-max", type=float, required=True, help="bounding box maximum longitude")
p.add("--ntimes",   type=int,   help="number of time points to process, starting from the first. Note that the CM2.6 dataset is daily, so this would be number of days. If unset, uses whole dataset.")
p.add("--co2-increase", action="store_true", help="use 1%% annual CO2 increase CM2.6 dataset. By default, uses control (no increase)")
p.add("--factor",   type=int,   required=True, help="resolution degradation factor")

p.add("--model-state-dict-file", type=str, required=True, help="model state dict file (*.pth)")
p.add("--device",  type=str, default="cuda", help="neural net device (e.g. cuda, cuda:0, cpu)")
p.add("--out-dir", type=str, required=True,  help="folder to save output dataset to")

p.add("--train-split", required=True)
p.add("--test-split",  required=True)
p.add("--batch_size",  required=True)

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

# store bounding box in a struct-like
bbox = BoundingBox(
        options.lat_min,  options.lat_max,
        options.long_min, options.long_max)
if not bounding_box.validate_nonempty(bbox):
    cli.fail(2, f"provided bounding box describes an empty region: {bbox}")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("retrieving CM2.6 dataset via Pangeo Cloud Datastore...")
surface_fields, _grid = lib.retrieve_cm2_6(options.pangeo_catalog_uri, options.co2_increase)

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
