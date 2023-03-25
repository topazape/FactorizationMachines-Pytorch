from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import train_test_split

from factorization_machines.utils import get_logger


class PreprocessCriteo:
    """Preprocess Criteo dataset."""

    def __init__(self, filepath: Path) -> None:
        """Initialize.

        Args:
            filepath (Path): filepath

        Raises:
            FileNotFoundError: if file not found
        """
        self.filepath = filepath

        self.logger = get_logger()
        self.logger.info("Preprocess Criteo dataset")

        self.category2index: dict[str, int] | None = None

    @property
    def target_column(self) -> str:
        """Target column."""
        return "label"

    @property
    def numerical_columns(self) -> list[str]:
        """Numerical columns."""
        return [f"numerical_feat_{i}" for i in range(13)]

    @property
    def categorical_columns(self) -> list[str]:
        """Categorical columns."""
        return [f"categorical_feat_{i}" for i in range(26)]

    @property
    def input_numerical_dim(self) -> int:
        """Input numerical dimension."""
        return len(self.numerical_columns)

    @property
    def input_categorical_dim(self) -> int:
        """Input categorical dimension."""
        if self.category2index is None:
            return 0
        return len(self.category2index)

    def load_data(self, data_size_rate: float, seed: int) -> pl.DataFrame:
        """Load data.

        Args:
            data_size_rate (float): data size rate
            seed (int): random seed

        Returns:
            (pl.DataFrame): dataframe
        """
        self.logger.info("loading...")
        column_names = [
            self.target_column,
            *self.numerical_columns,
            *self.categorical_columns,
        ]
        data = pl.read_csv(self.filepath, has_header=False, separator="\t")
        data.columns = column_names

        self.logger.info("sampling...")
        data = data.sample(frac=data_size_rate, seed=seed)

        return data

    def train_test_split(
        self,
        data: pl.DataFrame,
        test_size_rate: float,
        shuffle: bool,
        seed: int,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Train test split.

        Args:
            data (pd.DataFrame): dataframe
            test_size_rate (float): test size rate
            shuffle (bool): shuffle or not
            seed (int): random seed

        Returns:
            (tuple[pd.DataFrame, pd.DataFrame]): train and test dataframe
        """
        self.logger.info("train test split...")

        train_df, test_df = train_test_split(
            data,
            test_size=test_size_rate,
            shuffle=shuffle,
            random_state=seed,
        )

        return (train_df, test_df)

    def fit_categorical(self, data: pl.DataFrame) -> None:
        """Fit categorical columns.

        Args:
            data (pl.DataFrame): categorical dataframe
        """
        _categories = data.columns
        categories = np.unique(data.to_numpy().flatten()).tolist()
        categories = categories + _categories
        self.category2index = {cat: idx for idx, cat in enumerate(categories)}

    def transform_categorical(self, data: pl.DataFrame) -> pd.DataFrame:
        """Transform categorical columns.

        Args:
            data (pd.DataFrame): categorical dataframe

        Returns:
            (pd.DataFrame): transformed dataframe

        Raises:
            NotFittedError: if not fitted
        """
        if self.category2index is None:
            raise NotFittedError

        data = data.select(
            [
                (
                    pl.col(col)
                    .map_dict(self.category2index, default=self.category2index.get(col))
                    .keep_name()
                )
                for col in data.columns
            ]
        )
        return data

    def preprocess(self, data: pl.DataFrame) -> dict[str, pd.DataFrame]:
        """Preprocess.

        Args:
            data (pl.DataFrame): dataframe.

        Returns:
            (dict[str, pd.DataFrame]): preprocessed data
        """
        self.logger.info("preprocessing...")

        self.logger.info("preprocess for numerical columns...")
        n_cols = pl.col(self.numerical_columns)
        n_df = data.select(np.log(n_cols - n_cols.min() + 2).fill_null(0)).to_pandas()

        # categorical columns
        self.logger.info("preprocess for categorical columns...")
        c_df = data.select(self.categorical_columns).fill_null("nan")
        c_df = c_df.select(
            [(f"{col}_" + pl.col(col)).keep_name() for col in c_df.columns],
        )
        # transform to index
        if self.category2index is None:
            self.logger.info("fit categorical columns...")
            self.fit_categorical(c_df)

        c_df = self.transform_categorical(c_df)

        # y
        t_df = data.select(self.target_column).to_pandas()

        return {
            "numerical": n_df,
            "categorical": c_df,
            "label": t_df,
        }
