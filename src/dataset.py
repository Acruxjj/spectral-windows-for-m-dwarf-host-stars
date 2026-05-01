import os
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_spectra_parquets(folder_path: str) -> pd.DataFrame:
    """
    Concatenate every `.parquet` file under `folder_path`.
    """
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(
            f"Data directory not found: {folder_path}\n"
            "The parquet files are not tracked by git — upload them to this folder manually."
        )
    paths = [
        os.path.join(folder_path, name)
        for name in os.listdir(folder_path)
        if name.endswith(".parquet")
    ]
    if not paths:
        raise FileNotFoundError(f"No .parquet files found in {folder_path}")
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)


@dataclass
class SpectraSplits:
    """
    Train/val/test arrays plus the per-target StandardScaler list.
    """

    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train_scaled: np.ndarray
    y_val_scaled: np.ndarray
    y_test_scaled: np.ndarray
    output_scalers: list[StandardScaler]
    output_targets: tuple[str, ...]


def prepare_splits(
    df: pd.DataFrame,
    output_targets: Sequence[str],
    input_feature: str = "flux",
    test_size: float = 0.1,
    val_size: float = 0.1,
    random_state: int = 42,
) -> SpectraSplits:
    """
    Stack spectra, split, and fit a StandardScaler per target on the train set.
    """
    X = np.stack(df[input_feature].values)
    y = df[list(output_targets)].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=random_state
    )

    n_targets = len(output_targets)
    y_train_scaled = np.zeros_like(y_train, dtype=float)
    y_val_scaled = np.zeros_like(y_val, dtype=float)
    y_test_scaled = np.zeros_like(y_test, dtype=float)
    scalers: list[StandardScaler] = []

    y_train_np = y_train.to_numpy()
    y_val_np = y_val.to_numpy()
    y_test_np = y_test.to_numpy()

    for i in range(n_targets):
        scaler = StandardScaler()
        y_train_scaled[:, i] = scaler.fit_transform(
            y_train_np[:, i].reshape(-1, 1)
        ).ravel()
        y_val_scaled[:, i] = scaler.transform(y_val_np[:, i].reshape(-1, 1)).ravel()
        y_test_scaled[:, i] = scaler.transform(y_test_np[:, i].reshape(-1, 1)).ravel()
        scalers.append(scaler)

    return SpectraSplits(
        X_train=X_train[..., np.newaxis],
        X_val=X_val[..., np.newaxis],
        X_test=X_test[..., np.newaxis],
        y_train_scaled=y_train_scaled,
        y_val_scaled=y_val_scaled,
        y_test_scaled=y_test_scaled,
        output_scalers=scalers,
        output_targets=tuple(output_targets),
    )


def _to_named_dict(y: np.ndarray, targets: Sequence[str]) -> dict[str, np.ndarray]:
    return {name: y[:, i] for i, name in enumerate(targets)}


def _parse_function(x, y):
    x = tf.cast(x, tf.float32)
    y = {name: tf.cast(val, tf.float32) for name, val in y.items()}
    return x, y


def make_tf_datasets(
    splits: SpectraSplits,
    batch_size: int = 32,
    shuffle_buffer: int = 1000,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """
    Build `(train, val, test)` `tf.data.Dataset`s from a `SpectraSplits`.
    """
    targets = splits.output_targets

    train = tf.data.Dataset.from_tensor_slices(
        (splits.X_train, _to_named_dict(splits.y_train_scaled, targets))
    )
    train = (
        train.map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE)
        .shuffle(buffer_size=shuffle_buffer)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    val = tf.data.Dataset.from_tensor_slices(
        (splits.X_val, _to_named_dict(splits.y_val_scaled, targets))
    )
    val = (
        val.map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    test = tf.data.Dataset.from_tensor_slices(
        (splits.X_test, _to_named_dict(splits.y_test_scaled, targets))
    )
    test = (
        test.map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    return train, val, test
