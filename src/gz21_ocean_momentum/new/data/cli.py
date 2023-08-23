import gz21_ocean_momentum.new.data.step as step
from   gz21_ocean_momentum.new.common.bounding_box import BoundingBox

import configargparse

DESCRIPTION = "Read data from the CM2.6 and \
        apply coarse graining. Stores the resulting dataset into an MLFLOW \
        experiment within a specific run."

p = configargparse.ArgParser()
p.add("--config-file", is_config_file=True, help="config file path")
p.add("--out-dir",  type=str,   required=True)
p.add("--lat-min",  type=float, required=True)
p.add("--lat-max",  type=float, required=True)
p.add("--long-min", type=float, required=True)
p.add("--long-max", type=float, required=True)
p.add("--cyclize",  action="store_true", help="global data; make cyclic along longitude")
p.add("--ntimes",   type=int,   required=True, help="number of days (TODO)")
p.add("--co2-increase", action="store_true", help="use 1%% annual CO2 increase CM2.6 dataset. By default, uses control (no increase)")
p.add("--factor",   type=int, help="resolution degradation factor")

options = p.parse_args()

# form bounding box from input arguments
bounding_box = BoundingBox(
        options.lat_min,  options.lat_max,
        options.long_min, options.long_max)

CATALOG_URL = "https://raw.githubusercontent.com/\
pangeo-data/pangeo-datastore/\
master/\
intake-catalogs/master.yaml"

surface_fields, grid = step.download_cm2_6(CATALOG_URL, options.co2_increase)
forcings = step.preprocess_and_compute_forcings(
        grid, surface_fields, bounding_box, options.ntimes, options.cyclize,
        options.factor, "usurf", "vsurf")

# TODO: if path exists, gets a zarr.errors.ContainsGroupError
forcings.to_zarr(options.out_dir)
