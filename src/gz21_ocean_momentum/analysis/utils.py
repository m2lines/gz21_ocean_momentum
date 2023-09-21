# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 14:00:45 2020

@author: Arthur
"""
import numpy as np
import mlflow
from mlflow.tracking import client
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import pandas as pd
from gz21_ocean_momentum.analysis.analysis import TimeSeriesForPoint
import xarray as xr
from typing import Optional
from scipy.ndimage import gaussian_filter
from gz21_ocean_momentum.data.pangeo_catalog import get_patch, get_whole_data
from cartopy.crs import PlateCarree


from enum import Enum

CATALOG_URL = "https://raw.githubusercontent.com/pangeo-data/pangeo-datastore\
/master/intake-catalogs/master.yaml"


def correlation_map(truth: np.ndarray, pred: np.ndarray):
    """
    Return the correlation map.

    Parameters
    ----------
    truth : np.ndarray
        True values.
    pred : np.ndarray
        Predicted values

    Returns
    -------
    correlation_map : np.ndarray
        Correlation between true and predictions.

    """

    correlation_map = np.mean(truth * pred, axis=0)
    correlation_map -= np.mean(truth, axis=0) * np.mean(pred, axis=0)
    correlation_map /= np.std(truth, axis=0) * np.std(pred, axis=0)
    return correlation_map


def rmse_map(targets: np.ndarray, predictions: np.ndarray, normalized: bool = False):
    """Computes the rmse of the prediction time series at each point."""
    error = predictions - targets
    stds = np.std(targets, axis=0)
    if normalized:
        stds = np.clip(stds, np.min(stds[stds > 0]), np.inf)
    else:
        stds = 1
    rmse_map = np.sqrt(np.mean(np.power(error, 2), axis=0)) / stds
    return rmse_map


def download_data(run_id: str, rescale: bool = True) -> xr.Dataset:
    client_ = client.MlflowClient()
    data_file_name = client_.download_artifacts(run_id, "forcing")
    data = xr.open_zarr(data_file_name)
    data = data.rename({"xu_ocean": "longitude", "yu_ocean": "latitude"})
    if not rescale:
        return data
    data["S_x"] = data["S_x"] * 1e7
    data["S_y"] = data["S_y"] * 1e7
    return data


def download_pred(run_id: str, precision_to_std: bool = True) -> xr.Dataset:
    client_ = client.MlflowClient()
    pred_file_name = client_.download_artifacts(run_id, "test_output_0")
    pred = xr.open_zarr(pred_file_name)
    # For compatibility with old version
    if "S_xpred" in pred.keys():
        pred = pred.rename(S_xpred="S_x", S_ypred="S_y")
    if not precision_to_std:
        return pred
    pred["S_xscale"] = 1 / pred["S_xscale"]
    pred["S_yscale"] = 1 / pred["S_yscale"]
    return pred


def download_data_pred(
    run_id_data: str,
    run_id_pred: str,
    rescale_data: bool = True,
    precision_to_std_pred: bool = True,
) -> (xr.Dataset, xr.Dataset):
    data = download_data(run_id_data, rescale_data)
    pred = download_pred(run_id_pred, precision_to_std_pred)
    # Remove times from data which are not in predictions
    data = data.sel(time=slice(pred.time[0], pred.time[-1]))
    # Remove latitudes in data which are not in predictions
    data = data.sel(latitude=slice(pred["latitude"][0], pred["latitude"][-1]))
    return data, pred


def plot_time_series(
    data, pred, longitude: float, latitude: float, time: slice, std: bool = True
):
    plt.figure()
    xs = np.arange(time.start, time.stop, time.step)
    truth = (
        data["S_x"]
        .sel(longitude=longitude, latitude=latitude, method="nearest")
        .isel(time=time)
    )
    pred_mean = (
        pred["S_x"]
        .sel(longitude=longitude, latitude=latitude, method="nearest")
        .isel(time=time)
    )
    pred_std = (
        pred["S_xscale"]
        .sel(longitude=longitude, latitude=latitude, method="nearest")
        .isel(time=time)
    )
    plt.plot(xs, truth)
    plt.plot(xs, pred_mean)
    if std:
        plt.plot(xs, pred_mean + 1.96 * pred_std, "g--", linewidth=0.5)
        plt.plot(xs, pred_mean - 1.96 * pred_std, "g--", linewidth=0.5)
    plt.ylabel(r"$1e^{-7}m/s^2$")
    _ = plt.xlabel("days")


class DisplayMode(Enum):
    """Enumeration of the different display modes for viewing methods"""

    correlation = correlation_map
    rmse = rmse_map

    def diff_func(x, y):
        return np.mean(x - y, axis=0)

    difference = diff_func


def view_predictions(
    predictions: np.ndarray, targets: np.ndarray, display_mode=DisplayMode.correlation
):
    """Plots the correlation map for the passed predictions and targets.
    On clicking a point on the correlation map, the time series of targets
    and predictions at that point are shown in a new plot for further
    analysis."""
    # Compute the correlation map
    map_ = display_mode(targets, predictions)
    fig = plt.figure()
    plt.imshow(map_, origin="lower")
    plt.colorbar()
    plt.show()

    def onClick(event):
        time_series0 = TimeSeriesForPoint(predictions=predictions, truth=targets)
        time_series0.point = (int(event.xdata), int(event.ydata))
        time_series0.plot_pred_vs_true()

    fig.canvas.mpl_connect("button_press_event", onClick)


def sample(data: np.ndarray, step_time: int = 1, nb_per_time: int = 5, random_state: Optional[int] = None):
    """Samples points from the data, where it is assumed that the data
    is 4-D, with the first dimension representing time , the second
    the channel, and the others representing spatial dimensions.
    The sampling is done for every step_time image, and for each image
    nb_per_time points are randomly selected.

    Parameters
    ----------

    :data: ndarray, (n_time, n_channels, n_x, n_y)
        The time series of images to sample from.

    :step_time: int,
        The distance in time between two consecutive images used for the
        sampling.

    :nb_per_time: int,
        Number of points used (chosen randomly according to a uniform
        distribution over the spatial domain) for each image.
        
    :random_state: int, optional,
        Random state used for the random number generator.


    Returns
    -------
    :sample: ndarray, (n_time / step_time, n_channels, nb_per_time )
        The sampled data.
    """
    if data.ndim != 4:
        raise ValueError("The data is expected to have 4 dimensions.")
    np.random.seed(random_state)
    n_times, n_channels, n_x, n_y = data.shape
    time_indices = np.arange(0, n_times, step_time)
    x_indices = np.random.randint(0, n_x, (time_indices.shape[0], 2, nb_per_time))
    y_indices = np.random.randint(0, n_y, (time_indices.shape[0], 2, nb_per_time))
    channel_indices = np.zeros_like(x_indices)
    channel_indices[:, 1, :] = 1
    time_indices = time_indices.reshape((-1, 1, 1))
    time_indices = time_indices.repeat(2, axis=1)
    time_indices = time_indices.repeat(nb_per_time, axis=2)

    selection = time_indices, channel_indices, x_indices, y_indices
    sample = data[selection]
    return sample


def plot_dataset(dataset: xr.Dataset, plot_type=None, *args, **kargs):
    """
    Calls the plot function of each variable in the dataset.

    Parameters
    ----------
    dataset : xr.Dataset
        Dataset whose variables we wish to plot.
    plot_type : str, optional
        Plot type used for each variable in the dataset. The default is None.
    *args : list
        List of args passed on to the plot function.
    **kargs : dictionary
        Dictionary of args passed on to the plot function.

    Returns
    -------
    None.

    """
    plt.figure(figsize=(20, 5 * int(len(dataset) / 2)))
    kargs_ = [dict() for i in range(len(dataset))]

    def process_list_of_args(name: str):
        if name in kargs:
            if isinstance(kargs[name], list):
                for i, arg_value in enumerate(kargs[name]):
                    kargs_[i][name] = arg_value
            else:
                for i in range(len(dataset)):
                    kargs_[i][name] = kargs[name]
            kargs.pop(name)

    process_list_of_args("vmin")
    process_list_of_args("vmax")
    for i, variable in enumerate(dataset):
        plt.subplot(int(len(dataset) / 2), 2, i + 1)
        if plot_type is None:
            try:
                # By default we set the cmap to coolwarm
                kargs.setdefault("cmap", "coolwarm")
                dataset[variable].plot(*args, **kargs_[i], **kargs)
            except AttributeError:
                kargs.pop("cmap", None)
                dataset[variable].plot(*args, **kargs)
        else:
            plt_func = getattr(dataset[variable].plot, plot_type)
            plt_func(*args, **kargs)


def dataset_to_movie(dataset: xr.Dataset, interval: int = 50, *args, **kargs):
    """
    Generates animations for all the variables in the dataset

    Parameters
    ----------
    dataset : xr.Dataset
        Dataset used to generate movie. Must contain dimension 'time'.
    interval : int, optional
        Interval between frames in milliseconds. The default is 50.
    *args : list
        Positional args passed on to plot function.
    **kargs : dictionary
        keyword args passed on to plot function.

    Returns
    -------
    ani : TYPE
        Movie animation.

    """
    fig = plt.figure(figsize=(20, 5 * int(len(dataset) / 2)))
    axes = list()
    ims = list()
    for i, variable in enumerate(dataset.keys()):
        axes.append(fig.add_subplot(int(len(dataset) / 2), 2, i + 1))
    for i, t in enumerate(dataset["time"]):
        im = list()
        for axis, variable in zip(axes, dataset.keys()):
            plt.sca(axis)
            img = dataset[variable].isel(time=i).plot(*args, **kargs)
            cb = img.colorbar
            cb.remove()
            im.append(img)
        ims.append(im)
    ani = animation.ArtistAnimation(
        fig, ims, interval=interval, blit=True, repeat_delay=1000
    )
    return ani


def play_movie(predictions: np.ndarray, title: str = "", interval: int = 500):
    fig = plt.figure()
    ims = list()
    mean = np.mean(predictions)
    std = np.std(predictions)
    vmin, vmax = mean - std, mean + std
    for im in predictions:
        ims.append(
            [
                plt.imshow(
                    im,
                    vmin=vmin,
                    vmax=vmax,
                    cmap="YlOrRd",
                    origin="lower",
                    animated=True,
                )
            ]
        )
    ani = animation.ArtistAnimation(
        fig, ims, interval=interval, blit=True, repeat_delay=1000
    )
    plt.title(title)
    plt.show()
    return ani


class GlobalPlotter:
    """General class to make plots for global data. Handles masking of
    continental data + showing a band near coastlines."""

    def __init__(self, margin: int = 10, cbar: bool = True, ice: bool = True):
        self.mask = self._get_global_u_mask()
        self.margin = margin
        self.cbar = cbar
        self.ticks = dict(x=None, y=None)
        self.ice = ice

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, value):
        self._mask = value

    @property
    def borders(self):
        return self._borders

    @borders.setter
    def borders(self, value):
        self._borders = value

    @property
    def margin(self):
        return self._margin

    @margin.setter
    def margin(self, margin):
        self._margin = margin
        self.borders = self._get_continent_borders(self.mask, self.margin)

    @property
    def x_ticks(self):
        return self.ticks["x"]

    @x_ticks.setter
    def x_ticks(self, value):
        self.ticks["x"] = value

    @property
    def y_ticks(self):
        return self.ticks["y"]

    @y_ticks.setter
    def y_ticks(self, value):
        self.ticks["y"] = value

    def plot(
        self,
        u: xr.DataArray = None,
        projection_cls=PlateCarree,
        lon: float = -100.0,
        lat: float = None,
        ax=None,
        animated=False,
        borders_color="grey",
        borders_alpha=1.0,
        colorbar_label="",
        **plot_func_kw,
    ):
        """
        Plots the passed velocity component on a map, using the specified
        projection. Uses the instance's mask to set as nan some values.

        Parameters
        ----------
        u : xr.DataArray
            Velocity array. The default is None.
        projection : Projection
            Projection used for the 2D plot.
        lon : float, optional
            Central longitude. The default is -100.0.
        lat : float, optional
            Central latitude. The default is None.

        Returns
        -------
        None.

        """
        fig = plt.figure()
        projection = projection_cls(lon)
        if ax is None:
            ax = plt.axes(projection=projection)
        mesh_x, mesh_y = np.meshgrid(u["longitude"], u["latitude"])
        if u is not None:
            extra = self.mask.isel(longitude=slice(0, 10))
            extra["longitude"] = extra["longitude"] + 360
            mask = xr.concat((self.mask, extra), dim="longitude")
            mask = mask.interp({k: u.coords[k] for k in ("longitude", "latitude")})
            u = u * mask
            im = ax.pcolormesh(
                mesh_x,
                mesh_y,
                u.values,
                transform=PlateCarree(),
                animated=animated,
                **plot_func_kw,
            )
        if self.x_ticks is not None:
            ax.set_xticks(self.x_ticks)
        if self.y_ticks is not None:
            ax.set_yticks(self.y_ticks)
        ax.set_global()
        ax.coastlines()
        # "Gray-out" near continental locations
        if self.margin > 0:
            extra = self.borders.isel(longitude=slice(0, 10))
            extra["longitude"] = extra["longitude"] + 360
            borders = xr.concat((self.borders, extra), dim="longitude")
            borders = borders.interp(
                {k: u.coords[k] for k in ("longitude", "latitude")}
            )
            borders_cmap = colors.ListedColormap(
                [
                    borders_color,
                ]
            )
            ax.pcolormesh(
                mesh_x,
                mesh_y,
                borders,
                animated=animated,
                transform=PlateCarree(),
                alpha=borders_alpha,
                cmap=borders_cmap,
            )
        # Add locations of ice
        if self.ice:
            ice = self._get_ice_border()
            ice = xr.where(ice, 1.0, 0.0)
            ice = ice.interp({k: u.coords[k] for k in ("longitude", "latitude")})
            ice = xr.where(ice != 0, 1.0, 0.0)
            ice = abs(ice.diff(dim="longitude")) + abs(ice.diff(dim="latitude"))
            ice = xr.where(ice != 0.0, 1, np.nan)
            ice_cmap = colors.ListedColormap(
                [
                    "black",
                ]
            )
            ax.pcolormesh(
                mesh_x,
                mesh_y,
                ice,
                animated=animated,
                transform=PlateCarree(),
                alpha=0.5,
                cmap=ice_cmap,
            )
        if u is not None and self.cbar:
            cbar = plt.colorbar(im, shrink=0.6)
            if colorbar_label:
                cbar.set_label(colorbar_label)
        return ax

    @staticmethod
    def _get_global_u_mask(factor: int = 4, base_mask: xr.DataArray = None):
        """
        Return the global mask of the low-resolution surface velocities for
        plots. While the coarse-grained velocities might be defined on
        continental points due to the coarse-graining procedures, these are
        not shown as we do not use them -- the mask for the forcing is even
        more restrictive, as it removes any point within some margin of the
        velocities mask.

        Parameters
        ----------
        factor : int, optional
            Coarse-graining factor. The default is 4.

        base_mask: xr.DataArray, optional
            # TODO
            Not implemented for now.

        Returns
        -------
        None.

        """
        if base_mask is not None:
            mask = base_mask
        else:
            _, grid_info = get_whole_data(CATALOG_URL, 0)
            mask = grid_info["wet"]
            mask = mask.coarsen(dict(xt_ocean=factor, yt_ocean=factor))
        mask_ = mask.max()
        mask_ = mask_.where(mask_ > 0.1)
        mask_ = mask_.rename(dict(xt_ocean="longitude", yt_ocean="latitude"))
        return mask_.compute()

    @staticmethod
    def _get_ice_border():
        """Return an xarray.DataArray that indicates the locations of ice
        in the oceans."""
        temperature, _ = get_patch(CATALOG_URL, 1, None, 0, "surface_temp")
        temperature = temperature.rename(
            dict(xt_ocean="longitude", yt_ocean="latitude")
        )
        temperature = temperature["surface_temp"].isel(time=0)
        ice = xr.where(temperature <= 0.0, True, False)
        return ice

    @staticmethod
    def _get_continent_borders(base_mask: xr.DataArray, margin: int):
        """
        Returns a boolean xarray DataArray corresponding to a mask of the
        continents' coasts, which we do not process.
        Hence margin should be set according to the model.

        Parameters
        ----------
        mask : xr.DataArray
            Mask taking value 1 where coarse velocities are defined and used
            as input and nan elsewhere.
        margin : int
            Margin imposed by the model used, i.e. number of points lost on
            one side of a square.

        Returns
        -------
        mask : xr.DataArray
            Boolean DataArray taking value True for continents.

        """
        assert margin >= 0, "The margin parameter should be a non-negative" " integer"
        assert base_mask.ndim <= 2, "Velocity array should have two" " dims"
        # Small trick using the guassian filter function
        mask = xr.apply_ufunc(
            lambda x: gaussian_filter(x, 1.0, truncate=margin), base_mask
        )
        mask = np.logical_and(np.isnan(mask), ~np.isnan(base_mask))
        mask = mask.where(mask)
        return mask.compute()


def apply_complete_mask(array, pred, uv_plotter):
    mask = uv_plotter.borders
    mask2 = uv_plotter.mask
    mask = mask.interp({k: array.coords[k] for k in ["longitude", "latitude"]})
    mask2 = mask2.interp({k: array.coords[k] for k in ["longitude", "latitude"]})
    array = array.where(np.isnan(mask) & (~np.isnan(mask2)))
    array = array.sel(latitude=slice(pred["latitude"][0], pred["latitude"][-1]))
    return array


def plot_training_subdomains(
    run_id,
    global_plotter: GlobalPlotter,
    alpha=0.5,
    bg_variable=None,
    facecolor="blue",
    edgecolor=None,
    linewidth=None,
    fill=False,
    *plot_args,
    **plot_kwd_args,
):
    """
    Plots the training subdomains used for a given training run. Retrieves
    those subdomains from the run's parameters. Additionally, provide the
    latex code of a table with the latitudes and longitudes of each
    subdomain.

    Parameters
    ----------
    run_id : str
        Id of the training run.
    global_plotter : GlobalPlotter
        DESCRIPTION.
    alpha : TYPE, optional
        DESCRIPTION. The default is 0.5.
    facecolor : TYPE, optional
        DESCRIPTION. The default is 'blue'.
    edgecolor : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    None.

    """
    # First retrieve the run's data
    run = mlflow.get_run(run_id)
    run_params = run.data.params
    data_ids = run_params["source.run_id"].split("/")
    # retrieve the latex code for the table from file
    with open("analysis/latex_table.txt") as f:
        lines = f.readlines()
        latex_start = "".join(lines[:3])
        latex_line = lines[4]
        latex_end = "".join(lines[6:])
    latex_lines = []
    subdomain_names = "ABCDE"
    # Plot the map
    ax = global_plotter.plot(bg_variable, *plot_args, **plot_kwd_args)
    for i, data_id in enumerate(data_ids):
        # Recover the coordinates of the rectangular subdomain
        run = mlflow.get_run(data_id)
        run_params = run.data.params
        lat_min, lat_max = run_params["lat_min"], run_params["lat_max"]
        lon_min, lon_max = run_params["long_min"], run_params["long_max"]
        lat_min, lat_max = float(lat_min), float(lat_max)
        lon_min, lon_max = float(lon_min), float(lon_max)
        x, y = lon_min, lat_min
        width, height = lon_max - lon_min, lat_max - lat_min
        ax.add_patch(
            Rectangle(
                (x, y),
                width,
                height,
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=linewidth,
                fill=fill,
                alpha=alpha,
            )
        )
        # Add the table line
        lat_range = str(lat_min) + "\\degree, " + str(lat_max) + "\\degree"
        lon_range = str(lon_min) + "\\degree, " + str(lon_max) + "\\degree"
        latex_lines.append(latex_line.format(subdomain_names[i], lat_range, lon_range))
    latex_lines = "".join(latex_lines)
    latex = "".join((latex_start, latex_lines, latex_end))
    print(latex)
    plt.show()
    return ax


def anomalies(dataset: xr.Dataset, dim: str = "time.month"):
    """Returns a dataset of the anomalies."""
    grouped_data = dataset.groupby(dim)
    return grouped_data - grouped_data.mean()
