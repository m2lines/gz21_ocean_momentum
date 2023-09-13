import gz21_ocean_momentum.step.data.lib as lib
import gz21_ocean_momentum.common.cli as cli
from   gz21_ocean_momentum.common.bounding_box import BoundingBox

import gz21_ocean_momentum.step.data.coarsen as coarsen

import configargparse

import dask.diagnostics
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# up to date as of 2023-09-01
DEF_CATALOG_URI = "https://raw.githubusercontent.com/pangeo-data/pangeo-datastore/d684158e92fb3f3ad3b34e7dc5bba52b22a3ba80/intake-catalogs/ocean.yaml"

DESCRIPTION = "GZ21 data step: download CM2.6 dataset, apply coarse graining \
and generate forcings. Saves result to disk in zarr format."

p = configargparse.ArgParser(description=DESCRIPTION)
p.add("--config-file", is_config_file=True, help="config file path")
p.add("--out-dir",  type=str,   required=True, help="folder to save generated forcings to (in zarr format)" )
p.add("--lat-min",  type=float, required=True, help="bounding box minimum latitude")
p.add("--lat-max",  type=float, required=True, help="bounding box maximum latitude")
p.add("--long-min", type=float, required=True, help="bounding box minimum longitude")
p.add("--long-max", type=float, required=True, help="bounding box maximum longitude")
p.add("--cyclize",  action="store_true", help="global data; make cyclic along longitude")
p.add("--ntimes",   type=int,   help="number of time points to process, starting from the first. Note that the CM2.6 dataset is daily, so this would be number of days. If unset, uses whole dataset.")
p.add("--co2-increase", action="store_true", help="use 1%% annual CO2 increase CM2.6 dataset. By default, uses control (no increase)")
p.add("--factor",   type=int,   required=True, help="resolution degradation factor")
p.add("--pangeo-catalog-uri", type=str, default=DEF_CATALOG_URI, help="URI to Pangeo ocean dataset intake catalog file")
p.add("--verbose", action="store_true", help="be more verbose (e.g. display progress)")

options = p.parse_args()

if not cli.path_is_nonexist_or_empty_dir(options.out_dir):
    cli.fail(1, "--out-dir output directory is invalid",
                "if the directory exists, ensure it is empty")

# store bounding box in a struct-like
bounding_box = BoundingBox(
        options.lat_min,  options.lat_max,
        options.long_min, options.long_max)

logger.info("retrieving CM2.6 dataset via Pangeo Cloud Datastore...")
if options.co2_increase:
    logger.info("-> using 1% annual CO2 increase dataset")
else:
    logger.info("-> using control dataset (no annual CO2 increase)")

surface_fields, grid = lib.retrieve_cm2_6(options.pangeo_catalog_uri, options.co2_increase)

logger.info("selecting input data bounding box...")
surface_fields = surface_fields.sel(
    xu_ocean=slice(bounding_box.long_min, bounding_box.long_max),
    yu_ocean=slice(bounding_box.lat_min,  bounding_box.lat_max))
grid = grid.sel(
    xu_ocean=slice(bounding_box.long_min, bounding_box.long_max),
    yu_ocean=slice(bounding_box.lat_min,  bounding_box.lat_max))

if options.ntimes is not None:
    surface_fields = surface_fields.isel(time=slice(options.ntimes))

if options.verbose:
    dask.diagnostics.ProgressBar().register()

surface_fields = surface_fields[["usurf", "vsurf"]]

logger.info("computing forcings...")
#forcings = lib.preprocess_and_compute_forcings(
#        surface_fields, grid, options.cyclize,
#        options.factor)
forcings = coarsen.eddy_forcing(surface_fields, grid, options.factor)

logger.info("selecting forcing bounding box...")
forcings = forcings.sel(
    xu_ocean=slice(bounding_box.long_min, bounding_box.long_max),
    yu_ocean=slice(bounding_box.lat_min,  bounding_box.lat_max))

# TODO previously removed -- seems to not change output
for var in forcings:
    forcings[var].encoding = {}
forcings = forcings.chunk({"time": 1})

logger.info(f"writing forcings zarr to directory: {options.out_dir}")
forcings.to_zarr(options.out_dir)