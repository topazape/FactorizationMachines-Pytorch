from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from factorization_machines.utils import AverageMeter, get_logger


class Trainer:
    """Trainer class for training and evaluating model."""

    def __init__(
        self,
        epochs: int,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        criterion: Any,
        optimizer: Any,
        metric_fn: Any,
        device: str,
        save_dir: str,
    ) -> None:
        """Initialize Trainer class.

        Args:
            epochs (int): Number of epochs.
            train_loader (DataLoader): Training data loader.
            valid_loader (DataLoader): Validation data loader.
            criterion (Any): Loss function.
            optimizer (Any): Optimizer.
            metric_fn (Any): Metric function.
            device (str): Device to use.
            save_dir (str): Directory to save model.
        """
        self.epochs = epochs
        self.train_loader, self.valid_loader = train_loader, valid_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.metric_fn = metric_fn
        self.device = device
        self.save_dir = save_dir

        self.logger = get_logger(str(Path(self.save_dir).joinpath("log.txt")))
        self.best_loss = float("inf")

    def fit(self, model: nn.Module) -> None:
        """Fit model.

        Args:
            model (nn.Module): Model to fit.
        """
        for epoch in range(self.epochs):
            model.train()
            losses = AverageMeter("train_loss")
            metrics = AverageMeter("train_metric")

            with tqdm(self.train_loader, dynamic_ncols=True) as pbar:
                pbar.set_description(f"[Epoch {epoch + 1}/{self.epochs}")

                for tr_data in pbar:
                    numerical_x = tr_data[0].float().to(self.device)
                    categorical_x = tr_data[1].to(self.device)
                    label = tr_data[2].reshape(-1, 1).float().to(self.device)

                    self.optimizer.zero_grad()
                    y_pred = model(numerical_x, categorical_x)
                    loss = self.criterion(y_pred, label)
                    loss.backward()
                    self.optimizer.step()

                    losses.update(value=loss.item())

                    metrics.update(
                        value=self.metric_fn(
                            torch.squeeze(label).detach().cpu().numpy(),
                            torch.squeeze(y_pred).detach().cpu().numpy(),
                        )
                    )

                    pbar.set_postfix({"loss": losses.value})

            self.logger.info(
                f"(train) epoch: {epoch} loss: {losses.avg} metric: {metrics.avg}"
            )
            self.evaluate(model, epoch=epoch)

    @torch.no_grad()
    def evaluate(self, model: nn.Module, epoch: int | None = None) -> None:
        """Evaluate model.

        Args:
            model (nn.Module): Model to evaluate.
            epoch (int, optional): Epoch number. Defaults to None.
        """
        losses = AverageMeter("valid_loss")
        metrics = AverageMeter("valid_metric")

        for va_data in tqdm(self.valid_loader):
            numerical_x = va_data[0].float().to(self.device)
            categorical_x = va_data[1].to(self.device)
            label = va_data[2].reshape(-1, 1).float().to(self.device)

            y_pred = model(numerical_x, categorical_x)

            loss = self.criterion(y_pred, label)
            losses.update(value=loss.item())

            metrics.update(
                value=self.metric_fn(
                    torch.squeeze(label).detach().cpu().numpy(),
                    torch.squeeze(y_pred).detach().cpu().numpy(),
                )
            )

        self.logger.info(
            f"(vaid) epoch: {epoch} loss: {losses.avg} metric: {metrics.avg}"
        )

        if epoch is not None:
            if losses.avg <= self.best_loss:
                self.best_acc = losses.avg
                torch.save(model.state_dict(), Path(self.save_dir).joinpath("best.pth"))
