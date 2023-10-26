import configargparse

import logging

from gz21_ocean_momentum.models import submodels

DESCRIPTION = "GZ21 inference step."

p = configargparse.ArgParser(description=DESCRIPTION)
p.add("--config-file", is_config_file=True, help="config file path")
p.add("--model-state-dict-file", type=str, required=True, help="model state dict file (*.pth)")
p.add("--forcing-data-dir",      type=str, required=True, help="directory containing zarr-format forcing data")

p.add("--train-split", required=True)
p.add("--test-split",  required=True)
p.add("--test-split",  required=True)

options = p.parse_args()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

xr_dataset = xr.open_zarr(options.forcing_data_dir)

# TODO hardcode submodel, transformation
# unlikely for a CLI we need to provide dynamic code loading
# we can enable this in the "library" interface!
submodel = gz21_ocean_momentum.models.submodels
transformation = gz21_ocean_momentum.models.transforms.SoftPlusTransform
