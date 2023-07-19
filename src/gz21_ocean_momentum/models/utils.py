#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Utilities used within pytorch models."""
import importlib
import logging

import mlflow
import torch

# TODO raehik 2023-05-12: this is used in dynamic module loading. not sure I've
# done it correctly
import gz21_ocean_momentum.train as train
from gz21_ocean_momentum.utils import (pickle_artifact, select_experiment,
                                       select_run)


def load_model_cls(model_module_name: str, model_cls_name: str):
    """
    Dynamically load a neural network class (or a model) from a models module.

    Attributes
    ----------
    model_module_name : str
        module name where the class is defined
    model_cls_name : str
        class name of the model

    Returns
    -------
    model_cls: subclass of Module
        Class that defines the model
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
    Prompt user to select a model, load its parameters, and return it.

    Returns
    -------
    net : Module
        Trained neural network selected by the user.
    """
    # FIXME the experiment will depend on local implementation...
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
