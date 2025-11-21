"""
Hyperview1 DataModule for patch-wise multivariate regression.
"""

from typing import Any, Sequence, Optional
from pathlib import Path

import torch
from torch.utils.data import Subset
import albumentations as A
from torchgeo.datamodules import NonGeoDataModule

from src.datasets.hyperview_1 import Hyperview1NonGeo

class Hyperview1NonGeoDataModule(NonGeoDataModule):
    """NonGeo LightningDataModule for the Hyperview-1 challenge dataset."""

    def __init__(
        self,
        data_root: str,
        batch_size: int = 4,
        num_workers: int = 2,
        #crop_size: int = 11,
        resize_size: int = 128,
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

        # Albumentations CenterCrop transform
        self.transform = A.Compose([
            #A.CenterCrop(height=crop_size, width=crop_size),
            A.Resize(height=resize_size, width=resize_size)
        ])

    def setup(self, stage: Optional[str] = None) -> None:
        """Create train/val/test datasets."""
        if stage in ("fit", None):
            full_train_dataset = self.dataset_class( 
                data_root=self.data_root,
                split="train",
                bands=self.bands,
                transform=self.transform,
                target_mean=self.target_mean,
                target_std=self.target_std
            )
            
            # Create 80/20 train/val split
            val_size = int(0.2 * len(full_train_dataset))
            train_size = len(full_train_dataset) - val_size
            
            self.train_dataset, self.val_dataset = Subset(full_train_dataset, range(train_size)), Subset(full_train_dataset, range(train_size, len(full_train_dataset)))
            
            self.val_dataset = self.dataset_class(
                data_root=self.data_root,
                split="val",
                bands=self.bands,
                transform=self.transform,
                target_mean=self.target_mean,
                target_std=self.target_std
            )

        if stage in ("test", None):
            self.test_dataset = self.dataset_class(
                data_root=self.data_root,
                split="test",
                bands=self.bands,
                transform=self.transform
            )