import gz21_ocean_momentum.step.data.lib as lib
import gz21_ocean_momentum.common.cli as cli
from   gz21_ocean_momentum.common.bounding_box import BoundingBox
import gz21_ocean_momentum.common.bounding_box as bounding_box

import configargparse

import dask.diagnostics
import logging

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

if not cli.path_is_nonexist_or_empty_dir(options.out_dir):
    cli.fail(1, "--out-dir output directory is invalid",
                "if the directory exists, ensure it is empty")

# store bounding box in a struct-like
bbox = BoundingBox(
        options.lat_min,  options.lat_max,
        options.long_min, options.long_max)
if not bounding_box.validate_nonempty(bbox):
    cli.fail(2, f"provided bounding box describes an empty region: {bbox}")

logger.info("retrieving CM2.6 dataset via Pangeo Cloud Datastore...")
surface_fields, grid = lib.retrieve_cm2_6(options.pangeo_catalog_uri, options.co2_increase)

logger.debug("dropping irrelevant data variables...")
surface_fields = surface_fields[["usurf", "vsurf"]]

if options.ntimes is not None:
    logger.info(f"slicing {options.ntimes} time points...")
    surface_fields = surface_fields.isel(time=slice(options.ntimes))

logger.info("selecting input data bounding box...")
surface_fields = bounding_box.bound_dataset("yu_ocean", "xu_ocean", surface_fields, bbox)
grid = bounding_box.bound_dataset("yu_ocean", "xu_ocean", grid, bbox)

logger.debug("placing grid dataset into local memory...")
grid = grid.compute()

if options.cyclize:
    logger.info("making dataset cyclic along longitude...")
    logger.info("WARNING: may be nonfunctional or have poor performance")
    surface_fields = lib.cyclize(
            surface_fields, "xu_ocean", options.factor)
    grid = lib.cyclize(
            grid,           "xu_ocean", options.factor)

    logger.debug("rechunking along cyclized dimension...")
    surface_fields = surface_fields.chunk({"xu_ocean": -1})
    grid = grid.chunk({"xu_ocean": -1})

logger.info("computing forcings...")
forcings = lib.compute_forcings_cm2_6(surface_fields, grid, options.factor)

logger.info("selecting forcing bounding box...")
forcings = bound_dataset("yu_ocean", "xu_ocean", forcings, bbox)

logger.info(f"writing forcings zarr to directory: {options.out_dir}")
forcings.to_zarr(options.out_dir)
