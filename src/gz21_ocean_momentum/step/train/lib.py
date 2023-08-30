import torch

from gz21_ocean_momentum.models.models1 import FullyCNN
import gz21_ocean_momentum.models.transforms

from gz21_ocean_momentum.common.bounding_box import BoundingBox

def do_something() -> torch.nn.Module:
    net = FullyCNN()
    tf = transforms.SoftPlusTransform()
    tf.indices = criterion.precision_indices
    net.final_transformation = tf
    return net

def x_use_net(net: torch.nn.Module, weight_decay: float, ):
    optimizer = optim.Adam(params, lr=learning_rates[0], weight_decay=weight_decay)
    lr_scheduler = MultiStepLR(optimizer, list(learning_rates.keys())[1:], gamma=0.1)

def train_epoch(trainer, dataloader: torch.utils.data.DataLoader,
                optimizer, scheduler):
    # TODO remove clipping?
    train_loss = trainer.train_for_one_epoch(
        train_dataloader, optimizer, lr_scheduler, clip=1.0
    )
    test = trainer.test(test_dataloader)
    if test == "EARLY_STOPPING":
        print(test)
        break
    test_loss, metrics_results = test
    # Log the training loss
    print("Train loss for this epoch is ", train_loss)
    print("Test loss for this epoch is ", test_loss)

    for metric_name, metric_value in metrics_results.items():
        print(f"Test {metric_name} for this epoch is {metric_value}")
    mlflow.log_metric("train loss", train_loss, i_epoch)
    mlflow.log_metric("test loss", test_loss, i_epoch)
    mlflow.log_metrics(metrics_results)

def select_subdomains(ds: torch.utils.data.Dataset, sds: list[BoundingBox]) -> list[torch.utils.data.Dataset]:
    """TODO requires xu_ocean, yu_ocean"""
    return [ ds.sel(xu_ocean=slice(sd.long_min, sd.long_max),
                    yu_ocean=slice(sd.lat_min,  sd.lat_max)) for sd in sds ]

def xyz(ds: torch.utils.data.Dataset, tf) -> ?:
    # TODO this is a temporary fix to implement seasonal patterns
    # TODO previously did copy.deepcopy on transform...?
    print(tf)
    dsEdit = tf.fit_transform(ds)
    with ProgressBar(), TaskInfo("Computing dataset"):
        # Below line only for speeding up debugging
        # xr_dataset = xr_dataset.isel(time=slice(0, 1000))
        xr_dataset = xr_dataset.compute()
    print(xr_dataset)
    dataset = RawDataFromXrDataset(xr_dataset)
    dataset.index = "time"
    dataset.add_input("usurf")
    dataset.add_input("vsurf")
    dataset.add_output("S_x")
    dataset.add_output("S_y")
    # TODO temporary addition, should be made more general
    if submodel == "transform2":
        dataset.add_output("S_x_d")
        dataset.add_output("S_y_d")
    if submodel == "transform4":
        dataset.add_input("s_x_formula")
        dataset.add_input("s_y_formula")
    train_index = int(train_split * len(dataset))
    test_index = int(test_split * len(dataset))
    features_transform = ComposeTransforms()
    targets_transform = ComposeTransforms()
    transform = DatasetTransformer(features_transform, targets_transform)
    dataset = DatasetWithTransform(dataset, transform)
    # dataset = MultipleTimeIndices(dataset)
    # dataset.time_indices = [0, ]
    train_dataset = Subset_(dataset, np.arange(train_index))
    test_dataset = Subset_(dataset, np.arange(test_index, len(dataset)))
    train_datasets.append(train_dataset)
    test_datasets.append(test_dataset)
    datasets.append(dataset)
