import argparse
import random
from pathlib import Path

import tomllib
import torch
import torchinfo
from sklearn.metrics import roc_auc_score
from torch import nn, optim
from torch.utils.data import DataLoader

from factorization_machines import FactorizationMachines, Trainer
from factorization_machines.dataset import FMsDataset, PreprocessCriteo
from factorization_machines.utils import get_device


def make_parser() -> argparse.Namespace:
    """Make parser.

    Returns:
        argparse.Namespace: parser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_file", type=str, metavar="FILE", help="path to config file"
    )
    parser.add_argument(
        "--seed", default=0, type=int, help="seed for initializing training"
    )

    return parser.parse_args()


def set_seed(seed: int = 42) -> None:
    """Set seed.

    Args:
        seed (int): random seed
    """
    random.seed(seed)
    torch.manual_seed(seed)


def create_dataset(filename: str) -> tuple[FMsDataset, FMsDataset, int, int]:
    """Create dataset.

    Returns:
        tuple[FMsDataset, FMsDataset]: train dataset, test dataset
    """
    filepath = Path(filename)
    if not filepath.exists():
        raise FileNotFoundError(f"{filepath} is not found.")

    preprocessor = PreprocessCriteo(filepath)
    data = preprocessor.load_data(data_size_rate=0.1, seed=42)
    train_data, test_data = preprocessor.train_test_split(
        data, test_size_rate=0.1, shuffle=False, seed=42
    )
    train_dataset = preprocessor.preprocess(train_data)
    test_dataset = preprocessor.preprocess(test_data)

    return (
        FMsDataset(train_dataset),
        FMsDataset(test_dataset),
        preprocessor.input_numerical_dim,
        preprocessor.input_categorical_dim,
    )


if __name__ == "__main__":
    args = make_parser()
    device = get_device()

    config_file = Path(args.config_file)
    if not config_file.exists():
        raise FileNotFoundError(f"{config_file} is not found.")

    config_file_dir = str(config_file.resolve().parent)

    with config_file.open(mode="rb") as f:
        config = tomllib.load(f)

    set_seed(args.seed)

    filename = config["dataset"]["filename"]
    train_dataset, test_dataset, input_numerical_dim, input_categorical_dim = (
        create_dataset(filename)
    )

    model = FactorizationMachines(
        input_numerical_dim=input_numerical_dim,
        input_categorical_dim=input_categorical_dim,
        embedding_dim=config["model"]["embedding_dim"],
        k=config["model"]["k"],
    )

    print(model)
    torchinfo.summary(model)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["dataloader"]["train"]["batch_size"],
        shuffle=config["dataloader"]["train"]["shuffle"],
        drop_last=config["dataloader"]["train"]["drop_last"],
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["dataloader"]["test"]["batch_size"],
        shuffle=False,
        drop_last=False,
    )

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["trainer"]["learning_rate"],
        weight_decay=config["trainer"]["weight_decay"],
    )

    trainer = Trainer(
        epochs=config["trainer"]["epochs"],
        train_loader=train_loader,
        valid_loader=test_loader,
        criterion=criterion,
        metric_fn=roc_auc_score,
        optimizer=optimizer,
        device=device,
        save_dir=config_file_dir,
    )
    model.to(device)
    trainer.fit(model)
