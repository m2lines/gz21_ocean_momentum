#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import importlib
import logging
import torch
import mlflow
# TODO Does this need renaming to GZ-ocean-momentum.utils?
from subgrid.utils import select_experiment, select_run, pickle_artifact
import subgrid.train as train


def load_model_cls(model_module_name: str, model_cls_name: str):
    """
    Function Purpose?

    Parameters
    ----------
    model_module_name : str
        Description?
    model_cls_name : str
        Description?

    Returns
    -------

    """
    try:
        module = importlib.import_module(model_module_name)
        model_cls = getattr(module, model_cls_name)
    except ModuleNotFoundError as e:
        raise type(e)(
            "Could not retrieve the module in which the trained model \
                      is defined: "
            + str(e)
        )
    except AttributeError as e:
        raise type(e)("Could not retrieve the model's class. " + str(e))
    return model_cls


def select_and_load():
    """
    Prompts the user to select a model from those available, load its
    parameters and returns it.

    Parameters
    ----------
    input : TYPE?
        Description?

    Returns
    -------

    """
    models_experiment_id, _ = select_experiment("21")
    cols = [
        "metrics.test loss",
        "params.time_indices",
        "params.model_cls_name",
        "params.source.run_id",
        "params.submodel",
    ]
    model_run = select_run(
        cols=cols,
        experiment_ids=[
            models_experiment_id,
        ],
        default_selection=0,
    )
    model_module_name = model_run["params.model_module_name"]
    model_cls_name = model_run["params.model_cls_name"]
    loss_cls_name = model_run["params.loss_cls_name"]
    model_cls = load_model_cls(model_module_name, model_cls_name)
    # Load the loss function to retrieve the number of required outputs from the
    # nn.
    try:
        n_targets = 2
        criterion = getattr(train.losses, loss_cls_name)(n_targets)
    except AttributeError as e:
        raise type(e)(
            "Could not find the loss class used for training, ", loss_cls_name
        )
    net = model_cls(2, criterion.n_required_channels)
    transformation = pickle_artifact(model_run.run_id, "models/transformation")
    net.final_transformation = transformation

    # Load parameters of pre-trained model
    logging.info("Loading the neural net parameters")
    client = mlflow.tracking.MlflowClient()
    model_file = client.download_artifacts(model_run.run_id, "models/trained_model.pth")
    net.load_state_dict(torch.load(model_file))
    print(net)
    return net
