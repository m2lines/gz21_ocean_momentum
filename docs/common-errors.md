# Troubleshooting common errors
## `User project specified in the request is invalid.`
If when running the data processing step, you see an error message like this:

```
$ python src/gz21_ocean_momentum/cli/data.py --lat-min -80 --lat-max 80 --long-min -280 --long-max 80 --factor 4 --ntimes 1
Traceback (most recent call last):
  File "/home/user/workspace/gz21_ocean_momentum/src/gz21_ocean_momentum/cli/data.py", line 90, in <module>
    patch_data, grid_data = get_patch(
                            ^^^^^^^^^^
  File "/home/user/workspace/gz21_ocean_momentum/src/gz21_ocean_momentum/data/pangeo_catalog.py", line 65, in get_patch
    uv_data = source.to_dask()
              ^^^^^^^^^^^^^^^^
  File "/home/user/workspace/gz21_ocean_momentum/venv/lib/python3.11/site-packages/intake_xarray/base.py", line 69, in to_dask
    return self.read_chunked()
           ^^^^^^^^^^^^^^^^^^^
  File "/home/user/workspace/gz21_ocean_momentum/venv/lib/python3.11/site-packages/intake_xarray/base.py", line 44, in read_chunked
    self._load_metadata()
  File "/home/user/workspace/gz21_ocean_momentum/venv/lib/python3.11/site-packages/intake/source/base.py", line 283, in _load_metadata
    self._schema = self._get_schema()
                   ^^^^^^^^^^^^^^^^^^
  File "/home/user/workspace/gz21_ocean_momentum/venv/lib/python3.11/site-packages/intake_xarray/base.py", line 18, in _get_schema
    self._open_dataset()
  File "/home/user/workspace/gz21_ocean_momentum/venv/lib/python3.11/site-packages/intake_xarray/xzarr.py", line 46, in _open_dataset
    self._ds = xr.open_dataset(self.urlpath, **kw)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/user/workspace/gz21_ocean_momentum/venv/lib/python3.11/site-packages/xarray/backends/api.py", line 570, in open_dataset
    backend_ds = backend.open_dataset(
                 ^^^^^^^^^^^^^^^^^^^^^
  File "/home/user/workspace/gz21_ocean_momentum/venv/lib/python3.11/site-packages/xarray/backends/zarr.py", line 934, in open_dataset
    store = ZarrStore.open_group(
            ^^^^^^^^^^^^^^^^^^^^^
  File "/home/user/workspace/gz21_ocean_momentum/venv/lib/python3.11/site-packages/xarray/backends/zarr.py", line 450, in open_group
    zarr_group = zarr.open_consolidated(store, **open_kwargs)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/user/workspace/gz21_ocean_momentum/venv/lib/python3.11/site-packages/zarr/convenience.py", line 1299, in open_consolidated
    meta_store = ConsolidatedStoreClass(store, metadata_key=metadata_key)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/user/workspace/gz21_ocean_momentum/venv/lib/python3.11/site-packages/zarr/storage.py", line 2890, in __init__
    meta = json_loads(self.store[metadata_key])
                      ~~~~~~~~~~^^^^^^^^^^^^^^
  File "/home/user/workspace/gz21_ocean_momentum/venv/lib/python3.11/site-packages/zarr/storage.py", line 1397, in __getitem__
    return self.map[key]
           ~~~~~~~~^^^^^
  File "/home/user/workspace/gz21_ocean_momentum/venv/lib/python3.11/site-packages/fsspec/mapping.py", line 143, in __getitem__
    result = self.fs.cat(k)
             ^^^^^^^^^^^^^^
  File "/home/user/workspace/gz21_ocean_momentum/venv/lib/python3.11/site-packages/fsspec/asyn.py", line 121, in wrapper
    return sync(self.loop, func, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/user/workspace/gz21_ocean_momentum/venv/lib/python3.11/site-packages/fsspec/asyn.py", line 106, in sync
    raise return_result
  File "/home/user/workspace/gz21_ocean_momentum/venv/lib/python3.11/site-packages/fsspec/asyn.py", line 61, in _runner
    result[0] = await coro
                ^^^^^^^^^^
  File "/home/user/workspace/gz21_ocean_momentum/venv/lib/python3.11/site-packages/fsspec/asyn.py", line 433, in _cat
    raise ex
  File "/usr/lib/python3.11/asyncio/tasks.py", line 442, in wait_for
    return await fut
           ^^^^^^^^^
  File "/home/user/workspace/gz21_ocean_momentum/venv/lib/python3.11/site-packages/gcsfs/core.py", line 878, in _cat_file
    headers, out = await self._call("GET", u2, headers=head)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/user/workspace/gz21_ocean_momentum/venv/lib/python3.11/site-packages/gcsfs/core.py", line 430, in _call
    status, headers, info, contents = await self._request(
                                      ^^^^^^^^^^^^^^^^^^^^
  File "/home/user/workspace/gz21_ocean_momentum/venv/lib/python3.11/site-packages/decorator.py", line 221, in fun
    return await caller(func, *(extras + args), **kw)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/user/workspace/gz21_ocean_momentum/venv/lib/python3.11/site-packages/gcsfs/retry.py", line 114, in retry_request
    return await func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/user/workspace/gz21_ocean_momentum/venv/lib/python3.11/site-packages/gcsfs/core.py", line 423, in _request
    validate_response(status, contents, path, args)
  File "/home/user/workspace/gz21_ocean_momentum/venv/lib/python3.11/site-packages/gcsfs/retry.py", line 99, in validate_response
    raise ValueError(f"Bad Request: {path}\n{msg}")
ValueError: Bad Request: https://storage.googleapis.com/download/storage/v1/b/cmip6/o/GFDL_CM2_6%2Fcontrol%2Fsurface%2F.zmetadata?alt=media
User project specified in the request is invalid.
```

...then you probably haven't set up your GCP credentials properly. The dataset
used is stored in a GCP Requester Pays bucket, meaning you need to have a valid
API key connected to a billing account.

See `docs/data.md#requester-pays` for a fix.
