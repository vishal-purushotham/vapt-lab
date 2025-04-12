import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
import logging

logger = logging.getLogger(__name__)


def normalize_data(data, scaler=None, feature_range=(0, 1)):
    """Normalizes the data using MinMaxScaler.

    Args:
        data (np.ndarray): Data to normalize.
        scaler (MinMaxScaler, optional): Existing scaler to use. If None, a new one is fitted.
        feature_range (tuple, optional): Desired range of transformed data. Defaults to (0, 1).

    Returns:
        tuple: (normalized_data, scaler)
    """
    data = np.asarray(data, dtype=np.float32)
    if np.any(np.isnan(data)):
        logger.warning("NaN values found in data. Replacing with 0 before scaling.")
        data = np.nan_to_num(data)

    if scaler is None:
        scaler = MinMaxScaler(feature_range=feature_range)
        scaler.fit(data)
        logger.info("Fitted new MinMaxScaler.")
    else:
        logger.info("Using provided MinMaxScaler.")

    # Check if scaler is fitted
    if not hasattr(scaler, 'scale_') or scaler.scale_ is None:
       raise ValueError("Scaler is not fitted. Call fit() or provide a fitted scaler.")

    normalized_data = scaler.transform(data)
    logger.info(f"Data normalized to range {feature_range}.")

    return normalized_data, scaler


class SlidingWindowDataset(Dataset):
    """Creates a dataset of sliding windows from time series data."""
    def __init__(self, data, window_size, horizon=1):
        """
        Args:
            data (np.ndarray): The input time series data (n_samples, n_features).
            window_size (int): The number of time steps in each input window.
            horizon (int): The number of time steps to predict ahead. Defaults to 1.
        """
        self.data = data
        self.window_size = window_size
        self.horizon = horizon
        if len(data) <= window_size:
             raise ValueError(f"Data length ({len(data)}) must be greater than window size ({window_size}).")
        logger.info(f"Created SlidingWindowDataset with window_size={window_size}, horizon={horizon}")

    def __getitem__(self, index):
        # Check bounds to prevent IndexError
        if index + self.window_size + self.horizon > len(self.data):
            raise IndexError(f"Index {index} is out of bounds for data length {len(self.data)} with window {self.window_size} and horizon {self.horizon}")

        x = self.data[index : index + self.window_size]
        y = self.data[index + self.window_size : index + self.window_size + self.horizon]

        # Ensure y has the correct shape, especially if horizon=1
        if self.horizon == 1:
            y = y.squeeze(0) # Remove the time dimension if horizon is 1

        # Convert to PyTorch tensors
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        # The effective length is reduced by the window size and horizon
        return len(self.data) - self.window_size - self.horizon + 1


def create_data_loaders(dataset: SlidingWindowDataset, batch_size: int, val_split: float = 0.1, shuffle: bool = True, num_workers: int = 0):
    """Creates training, validation, and optionally test DataLoaders.

    Args:
        dataset (Dataset): The dataset to wrap (e.g., SlidingWindowDataset).
        batch_size (int): How many samples per batch to load.
        val_split (float): Proportion of the dataset to use for validation (0 to 1).
        shuffle (bool): Whether to shuffle the training data every epoch.
        num_workers (int): How many subprocesses to use for data loading.

    Returns:
        tuple: (train_loader, val_loader)
    """
    train_loader, val_loader = None, None

    dataset_size = len(dataset)
    indices = list(range(dataset_size))

    if val_split < 0 or val_split > 1:
        raise ValueError("val_split must be between 0 and 1.")

    if val_split == 0.0:
        logger.info(f"Creating train_loader with {dataset_size} samples.")
        if shuffle:
            np.random.shuffle(indices)
            train_sampler = SubsetRandomSampler(indices)
            train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
        else:
             train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    else:
        split = int(np.floor(val_split * dataset_size))
        if split == 0 or split == dataset_size:
             raise ValueError(f"Validation split {val_split} results in an empty train or validation set.")

        if shuffle:
            np.random.shuffle(indices)

        train_indices, val_indices = indices[split:], indices[:split]

        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)

        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
        val_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers)

        logger.info(f"Created train_loader with {len(train_indices)} samples.")
        logger.info(f"Created val_loader with {len(val_indices)} samples.")

    return train_loader, val_loader
