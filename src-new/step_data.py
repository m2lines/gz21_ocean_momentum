def run_data_step_cm2_6(
        catalog_url,
        bounding_box: Optional[BoundingBox],
        ntimes: Optional[int],
        cyclize: bool,
        factor: int
        ) -> xr.Dataset:
    """Run data step on CM2.6 dataset."""
    catalog = intake.open_catalog(catalog_url)
    grid = catalog.ocean.GFDL_CM2_6.GFDL_CM2_6_control_ocean_surface
    grid = grid.to_dask()
    if co2_increase:
        surface_fields = catalog.ocean.GFDL_CM2_6.GFDL_CM2_6_control_ocean_surface
    else:
        surface_fields = catalog.ocean.GFDL_CM2_6.GFDL_CM2_6_one_percent_ocean_surface
    surface_fields = surface_fields.to_dask()
    preprocess(grid, surface_fields, bounding_box, ntimes, cyclize, factor,
               "usurf", "vsurf")
