import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class FMsDataset(Dataset):
    """Dataset class for Factorization Machines."""

    def __init__(self, data: dict[str, pd.DataFrame]) -> None:
        """Initialize the dataset.

        Args:
            data (dict[str, np.ndarray]): Dictionary containing the numerical, categorical and label data.

        """
        self.numerical_data = data["numerical"].to_numpy()
        self.categorical_data = data["categorical"].to_numpy()
        self.label = data["label"].to_numpy().reshape(-1, 1)

    def __len__(self) -> int:
        return len(self.label)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            self.numerical_data[index, :],
            self.categorical_data[index, :],
            self.label[index],
        )
