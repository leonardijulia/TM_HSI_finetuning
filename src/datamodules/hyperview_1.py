"""
Hyperview1 DataModule for patch-wise multivariate regression.
"""

from typing import Any, Sequence, Optional
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, random_split
import albumentations as A
from torchgeo.datamodules import NonGeoDataModule

from src.datasets.hyperview_1 import Hyperview1NonGeo

class SubsetDatasetWrapper(Dataset):
    """Wrap a dataset and only expose a subset of indices."""
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)
    
    def plot(self, *args, **kwargs):
        # Forward plot calls to the underlying dataset
        return self.dataset.plot(*args, **kwargs)

class Hyperview1NonGeoDataModule(NonGeoDataModule):
    """NonGeo LightningDataModule for the Hyperview-1 challenge dataset."""

    def __init__(
        self,
        data_root: str,
        batch_size: int = 4,
        num_workers: int = 2,
        stats_path: str = "/leonardo/home/userexternal/jleonard/experiments/data/statistics/hyperview_1",
        resize_size: int = 224,
        bands: str = "s2l2a",
        target_mean: Optional[Sequence[float]] = None,
        target_std: Optional[Sequence[float]] = None,
    ) -> None:
        """
        Args:
            data_root: Root directory of the dataset.
            batch_size: Batch size for DataLoaders.
            num_workers: Number of workers for DataLoaders.
            crop_size: CenterCrop size for patches.
            bands: List of band names to select.
            band_indices: List of band indices corresponding to `bands`.
            target_mean: Optional precomputed mean of targets.
            target_std: Optional precomputed std of targets.
        """
        # Pass dataset class and loader params to parent
        super().__init__(dataset_class=Hyperview1NonGeo, batch_size=batch_size, num_workers=num_workers)

        self.data_root = Path(data_root)
        self.bands = bands
        self.target_mean = target_mean
        self.target_std = target_std
        self.stats_path = Path(stats_path)

        # Albumentations CenterCrop transform
        self.transform = A.Compose([
            A.D4(),
            A.Resize(height=resize_size, width=resize_size)
        ])

    def setup(self, stage: Optional[str] = None) -> None:
        """Create train/val/test datasets."""
        if stage in ["fit", "validate"]:
            full_train_dataset = self.dataset_class( 
                data_root=self.data_root,
                split="train",
                bands=self.bands,
                transform=self.transform,
                stats_path=self.stats_path,
                target_mean=self.target_mean,
                target_std=self.target_std
            )
            
            # Create 80/20 train/val split
            val_size = int(0.2 * len(full_train_dataset))
            train_size = len(full_train_dataset) - val_size
            
            train_indices = list(range(train_size))
            val_indices = list(range(train_size, len(full_train_dataset)))

            # Wrap the subsets so we keep dataset methods like .plot()
            self.train_dataset = SubsetDatasetWrapper(full_train_dataset, train_indices)
            self.val_dataset = SubsetDatasetWrapper(full_train_dataset, val_indices)
            
              
        if stage in ["predict"]:
            self.predict_dataset = self.dataset_class(
                data_root=self.data_root,
                split="test",
                bands=self.bands,
                transform=self.transform,
            )