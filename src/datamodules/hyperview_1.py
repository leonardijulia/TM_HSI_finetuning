"""
Hyperview1 DataModule for patch-wise multivariate regression.
"""
from typing import Sequence, Optional
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import random_split
import albumentations as A
from torchgeo.datamodules import NonGeoDataModule
from src.datasets.hyperview_1 import Hyperview1NonGeo

class Hyperview1NonGeoDataModule(NonGeoDataModule):
    """NonGeo LightningDataModule for the Hyperview-1 challenge dataset."""

    def __init__(
        self,
        data_root: str,
        stats_path: str = "./data/statistics/hyperview_1",
        batch_size: int = 4,
        num_workers: int = 2,
        resize_size: int = 224,
        bands: str = "s2l2a",
        band_selection: str = "naive",
        srf_weight_file: Optional[str] = "SRF_S2L2A_hyperview_1_W.npy",
        target_mean: Optional[Sequence[float]] = None,
        target_std: Optional[Sequence[float]] = None,
    ) -> None:
        """
        Args:
            data_root: Root directory of the dataset.
            stats_path: Rood directory of the statistics.
            batch_size: Batch size for DataLoaders.
            num_workers: Number of workers for DataLoaders.
            resize_size: CenterCrop size for patches.
            bands: List of band names to select.
            band_selection: Strategy of band selection.
            srf_weight_file: the SRF weight matrix used foe the srf_grouping band selection strategy.
            target_mean: Optional precomputed mean of targets.
            target_std: Optional precomputed std of targets.
        """
        super().__init__(Hyperview1NonGeo, batch_size, num_workers)

        self.data_root = Path(data_root)
        self.stats_path = Path(stats_path)
        self.bands = bands
        self.band_selection = band_selection
        self.resize_size = resize_size
        
        if srf_weight_file:
            self.srf_weight_path = self.stats_path / srf_weight_file
        else:
            self.srf_weight_path = None
            
        self.manual_target_mean = target_mean
        self.manual_target_std = target_std
        self.target_mean = None
        self.target_std = None
            
        self._init_transforms()
        
    def _init_transforms(self):
            
        self.train_transform = A.Compose([
            A.D4(),
            A.Resize(height=self.resize_size, width=self.resize_size),
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32, min_holes=1, 
                            min_height=16, min_width=16, fill_value=0, p=0.2),
            A.GaussNoise(var_limit=(0.001, 0.01), mean=0, per_channel=True, p=0.2)
        ])
            
        self.val_transform = A.Compose([
            A.Resize(height=self.resize_size, width=self.resize_size)
        ])

    def setup(self, stage: Optional[str] = None) -> None:
        """Create train/val/test datasets."""
        
        if stage in ["fit", "validate", "val"]:
            
            full_dataset = self.dataset_class(
                root=self.data_root,
                split="train",
                stats_path=self.stats_path,
                bands=self.bands, # minimal init
            )
            
            val_size = int(0.2 * len(full_dataset))
            train_size = len(full_dataset) - val_size
            generator = torch.Generator().manual_seed(42)
            train_subset, val_subset = random_split(full_dataset, [train_size, val_size], generator=generator)

            if self.manual_target_mean is not None:
                self.target_mean = self.manual_target_mean
                self.target_std = self.manual_target_std
            else:
                train_labels = [full_dataset.samples[i]["label"] for i in train_subset.indices]
                train_labels = np.stack(train_labels)
                self.target_mean = train_labels.mean(axis=0)
                self.target_std = train_labels.std(axis=0) + 1e-4
            
            self.train_dataset = self.dataset_class(
                root=self.data_root,
                split="train",
                bands=self.bands,
                band_selection=self.band_selection,
                srf_weight_matrix=self.srf_weight_path,
                stats_path=self.stats_path,
                target_mean=self.target_mean,
                target_std=self.target_std,
                subset_idx=train_subset.indices,
                transform=self.train_transform
            )
            
            self.val_dataset = self.dataset_class(
                root=self.data_root,
                split="train", # It's technically from the 'train' folder
                bands=self.bands,
                band_selection=self.band_selection,
                srf_weight_matrix=self.srf_weight_path,
                stats_path=self.stats_path,
                target_mean=self.target_mean,
                target_std=self.target_std,
                subset_idx=val_subset.indices,
                transform=self.val_transform
            )

        if stage == "predict":
            self.predict_dataset = self.dataset_class(
                root=self.data_root,
                split="test",
                bands=self.bands,
                band_selection=self.band_selection,
                srf_weight_matrix=self.srf_weight_path,
                stats_path=self.stats_path,
                transform=self.val_transform
            )