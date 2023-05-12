#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 12:50:18 2020

@author: arthur
"""

import mlflow
from mlflow.tracking import client
import pandas as pd
import pickle
import models # local module. any nicer syntax here? (not .models)
import sys

sys.modules["models"] = models # what are we doing here anyway


class TaskInfo:
    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        print(f"Starting task: {self.name}")

    def __exit__(self, *args):
        print(f"Task completed: {self.name}")


def select_experiment(default_selection: str = ""):
    """
    Prompt user to select an experiment among all experiments in store. Return
    the name of the selected experiment.

    Returns
    -------
    str
        Name of the experiment selected by the user.

    """
    client_ = client.MlflowClient()
    list_of_exp = client_.list_experiments()
    dict_of_exp = {exp.experiment_id: exp.name for exp in list_of_exp}
    for id_, name in dict_of_exp.items():
        print(id_, ": ", name)
    selection = input("Select the id of an experiment: ") or default_selection
    return selection, dict_of_exp[selection]


def select_run(
    sort_by=None, cols=None, merge=None, default_selection: str = "", *args, **kargs
) -> object:
    """
    Allows to select a run from the tracking store interactively.

    Parameters
    ----------
    sort_by : str, optional
        Name of the column used for sorting the returned runs.
        The default is None.
    cols : list[str], optional
        List of column names printed to user. The default is None.
    merge : list of length-3 tuples, optional
        Describe how to merge information with other experiments.
        Each element of the list is a tuple
        (experiment_name, key_left, key_right), according to which the
        initial dataframe of runs will be merged with that corresponding
        to experiment_name, using key_left (from the first dataframe) and
        key_right (from the second dataframe).
    *args : list
        List of args passed on to mlflow.search_runs.
    **kargs : dictionary
        Dictionary of args passed on to mlflow.search_runs. In particular
        one may want to specify experiment_ids to select runs from a given
        list of experiments.

    Raises
    ------
    Exception
        DESCRIPTION.

    Returns
    -------
    pandas.Series
        Series describing the interactively selected run.

    """
    mlflow_runs = mlflow.search_runs(*args, **kargs)
    if cols is None:
        cols = list()
    cols = ["run_id", "experiment_id"] + cols
    if sort_by is not None:
        mlflow_runs.sort_values(by=sort_by)
        cols.append(sort_by)
    # Remove possible duplicate columns
    new_cols = list()
    for e in cols:
        if e not in new_cols:
            new_cols.append(e)
    cols = new_cols
    if merge is not None:
        for name, key_left, key_right in merge:
            experiment = mlflow.get_experiment_by_name(name)
            df2 = mlflow.search_runs(experiment_ids=experiment.experiment_id)
            mlflow_runs = pd.merge(
                mlflow_runs,
                df2,
                left_on=key_left,
                right_on=key_right,
                suffixes=("", "y"),
            )
    if len(mlflow_runs) == 0:
        raise Exception(
            "No data found. Check that you correctly set \
                        the store"
        )
    print(mlflow_runs[cols])
    id_ = int(input("Run id?") or default_selection)
    if id_ < 0:
        return 0
    return mlflow_runs.loc[id_, :]


def pickle_artifact(run_id: str, path: str):
    client = mlflow.tracking.MlflowClient()
    file = client.download_artifacts(run_id, path)
    f = open(file, "rb")
    return pickle.load(f)
