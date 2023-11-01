import configargparse

import logging

from gz21_ocean_momentum.train.base import Trainer

# TODO hardcode submodel, transformation
# unlikely for a CLI we need to provide dynamic code loading -- let's just give
# options
# we could enable such "dynamic loading" in the "library" interface!-- but, due
# to the class-based setup, it's a little complicated for a user to come in with
# their own code for some of these, and it needs documentation. so a task for
# later
import gz21_ocean_momentum.models.models1.FullyCNN as model_cls
import gz21_ocean_momentum.models.submodels.transform3 as submodel
import gz21_ocean_momentum.models.transforms.SoftPlusTransform as transformation

DESCRIPTION = "GZ21 inference step."

p = configargparse.ArgParser(description=DESCRIPTION)
p.add("--config-file", is_config_file=True, help="config file path")
p.add("--model-state-dict-file", type=str, required=True, help="model state dict file (*.pth)")
p.add("--forcing-data-dir",      type=str, required=True, help="directory containing zarr-format forcing data")
p.add("--device",  type=str, default="cuda", help="neural net device (e.g. cuda, cuda:0, cpu)")
p.add("--out-dir", type=str, required=True,  help="folder to save output dataset to")

p.add("--train-split", required=True)
p.add("--test-split",  required=True)
p.add("--batch_size",  required=True)

options = p.parse_args()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

cli.fail_if_path_is_nonempty_dir(
        1, f"--out-dir \"{options.out_dir}\" invalid", options.out_dir)

xr_dataset = xr.open_zarr(options.forcing_data_dir)

# convert forcings xarray to PyTorch dataset
# TODO identical loading code in train step (originally trainScript.py)
dataset = RawDataFromXrDataset(xr_dataset)
dataset.index = "time"
dataset.add_input("usurf")
dataset.add_input("vsurf")
dataset.add_output("S_x")
dataset.add_output("S_y")

# Load some extra parameters of the model.
# TODO allow general time_indices
time_indices = [
    0,
]
train_split = float(model_run["params.train_split"])
test_split = float(model_run["params.test_split"])
batch_size = batch_size if batch_size else int(model_run["params.batchsize"])
source_data_id = model_run["params.source.run-id"]
loss_cls_name = model_run["params.loss_cls_name"]
learning_rates = learning_rates_from_string(model_run["params.learning_rate"])
submodel_name = model_run["params.submodel"]

# Set up training criterion and select parameters to train
try:
    n_targets = dataset.n_targets
    criterion = getattr(losses, loss_cls_name)(n_targets)
except AttributeError as e:
    raise type(e)("Could not find the loss class used for training, ", loss_cls_name)

# load, prepare pre-trained neural net
net = model_cls(dataset.n_features, criterion.n_required_channels)
net.load_state_dict(torch.load(options.model_state_dict_file))
print(net)
#net.cpu() # TODO why needed?
dataset.add_transforms_from_model(net)

print("Size of training data: {}".format(len(train_dataset)))
print("Size of validation data : {}".format(len(test_dataset)))
print("Input height: {}".format(train_dataset.height))
print("Input width: {}".format(train_dataset.width))
print(train_dataset[0][0].shape)
print(train_dataset[0][1].shape)
print("Features transform: ", transform.transforms["features"].transforms)
print("Targets transform: ", transform.transforms["targets"].transforms)

# Net to GPU
with TaskInfo("Put neural network on device"):
    net.to(options.device)

print("width: {}, height: {}".format(dataset.width, dataset.height))

# Training itself
if n_epochs > 0:
    with TaskInfo("Training"):
        trainer = Trainer(net, options.device)
        trainer.criterion = criterion
        # Register metrics
        for metric_name, metric in metrics.items():
            trainer.register_metric(metric_name, metric)
        parameters = net.parameters()
        optimizer = torch.optim.Adam(parameters, lr=learning_rate)
        for i_epoch in range(n_epochs):
            train_loss = trainer.train_for_one_epoch(train_dataloader, optimizer)
            test_loss, metrics_results = trainer.test(test_dataloader)
            print("Epoch {}".format(i_epoch))
            print("Train loss for this epoch is {}".format(train_loss))
            print("Test loss for this epoch is {}".format(test_loss))

    with TaskInfo("Validation"):
        train_loss, train_metrics_results = trainer.test(train_dataloader)
        print(f"Final train loss is {train_loss}")

# Test
with ProgressBar(), TaskInfo("Create output dataset"):
    out = create_large_test_dataset(net, criterion, partition, loaders, options.device)
    ProgressBar().register()
    print("Start of actual computations...")
    out = out.chunk(dict(time=32))
    out.to_zarr(file_path)
    mlflow.log_artifact(file_path)
    print(f"Size of output data is {out.nbytes/1e9} GB")
