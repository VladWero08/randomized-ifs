"""
Files containting functions that generate toy datasets with different type of outliers:
scatted and clustered, used for comparing different isolation forest algorithms.
"""

import numpy as np
from sklearn.datasets import make_blobs


def generate_even_scattered_anomalies(n_samples: int = 500) -> np.ndarray:
    """
    Generates evenly distributed blobs:
    - blob 1: 50% data points, std = 1
    - blob 2: 50% data points, std = 1
    """
    X_train, _ = make_blobs(
        n_samples=n_samples,
        n_features=2,
        centers=[(10, 0), (0, 10)],
        cluster_std=1
    )

    return X_train


def generate_uneven_scatter_anomalies(n_samples: int = 500) -> np.ndarray:
    """
    Generates unevenly distributed blobs:
    - blob 1: 80% data points, std = 1.5
    - blob 2: 20% data points, std = 1
    """
    X_train, _ = make_blobs(
        n_samples=[int(n_samples * 0.8), int(n_samples * 0.2)],
        n_features=2,
        centers=[(10, 0), (0, 10)],
        cluster_std=[1, 1]
    )

    return X_train


def generate_clustered_anomalies(n_samples: int = 500) -> np.ndarray:
    """
    Generate multiple blobs, some of them being smaller 
    with a narrow standard deviation:
    - blob 1: 5% data points,   std = 1
    - blob 2: 40% data points,  std = 1.5
    - blob 3: 5% data points,   std = 1
    - blob 4: 50% data points,  std = 1.5
    """
    X_train, _ = make_blobs(
        n_samples=[int(n_samples * 0.05), int(n_samples * 0.4), int(n_samples * 0.05), int(n_samples * 0.5)],
        n_features=2,
        centers=[(0, 0), (10, 0), (10, 10), (0, 10)],
        cluster_std=[1, 1.5, 1, 1.5]
    )

    return X_train


def generate_sin_anomalies(n_samples: int = 1000) -> np.ndarray:
    """
    Generate a sinus wave as the dataset.
    """
    x = np.linspace(-10, 10, num=n_samples)
    y = 2.5 * np.sin(x) + np.random.normal(loc=0, scale=0.5, size=n_samples)
    X_train = np.column_stack((x, y))

    return X_train
