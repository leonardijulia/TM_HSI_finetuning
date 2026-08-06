from typing import Any, Union

import numpy as np
import torch
from torch import Tensor
import kornia.augmentation as K
from torchgeo.transforms import AugmentationSequential
from torchgeo.datamodules.geo import NonGeoDataModule
from torchgeo.samplers.utils import _to_tuple
from torchgeo.datamodules.utils import MisconfigurationException

from datasets.enmap_corine import EnMAPCorineDataset
from transforms.normalize import NormalizeMeanStd  

class EnMAPCorineDataModule(NonGeoDataModule):
    """LightningDataModule implementation for the EnMAP CORINE dataset."""

    def __init__(
        self,
        batch_size: int = 64,
        patch_size: Union[int, tuple[int, int]] = 128,
        num_workers: int = 0,
        band_selection: str = "naive",
        stats_path: str = "data/statistics",
        srf_weight_file: Optional[str] = "SRF_S2L2A_EnMAP_W.npy",
        **kwargs: Any,
    ) -> None:
        """Initialize the EnMAP Corine Benchmark DataModule.

        Args:
            batch_size: Size of each mini-batch.
            patch_size: Size of each patch, either an integer or a tuple (height, width).
            num_workers: Number of workers for parallel data loading.
            stats_path: Path to the directory containing normalization statistics (mu.npy and sigma.npy).
            **kwargs: Additional keyword arguments passed to the dataset.
        """
        super().__init__(EnMAPCorineDataset, batch_size, num_workers, **kwargs)

        self.patch_size = _to_tuple(patch_size)
        self.srf_weight_path = Path(stats_path, srf_weight_file)
        self.band_selection = band_selection
        
        kwargs["band_selection"] = band_selection
        kwargs["srf_weight_matrix"] = self.srf_weight_path

        try:
            mean = torch.tensor(np.load(f"{stats_path}/mu.npy"))
            std = torch.tensor(np.load(f"{stats_path}/sigma.npy"))
        except FileNotFoundError:
            raise MisconfigurationException("Missing statistics! Ensure mu.npy and sigma.npy are available.")

        if self.band_selection == "naive":
            mean = raw_mean[self.indices]
            std = raw_std[self.indices]
                    
        elif self.band_selection == "srf_grouping":
            if self.srf_weight_path is None:
                raise MisconfigurationException("SRF grouping requires srf_weight_path!")
            weights = torch.tensor(np.load(self.srf_weight_path)).float()
            mean = torch.matmul(raw_mean, weights)
                    
            # Std: Error propagation (assuming independence) -> var_new = var_old @ W^2
            raw_var = raw_std ** 2
            weights_sq = weights ** 2
            var = torch.matmul(raw_var, weights_sq)
            std = torch.sqrt(var)
                    
        else:
            raise ValueError(f"Unknown band selection: {band_selection}")

        # Define data augmentations
        self.train_aug = AugmentationSequential(
            K.Resize(_to_tuple(self.patch_size)),
            K.RandomResizedCrop(self.patch_size, scale=(0.6, 1.0)),
            K.RandomVerticalFlip(p=0.5),
            K.RandomHorizontalFlip(p=0.5),
            NormalizeMeanStd(mean=mean, std=std),
            data_keys=["image"],
        )

        self.val_aug = AugmentationSequential(
            K.Resize(_to_tuple(self.patch_size)),
            K.CenterCrop(self.patch_size),
            NormalizeMeanStd(mean=mean, std=std),
            data_keys=["image"],
        )

        self.test_aug = AugmentationSequential(
            K.Resize(_to_tuple(self.patch_size)),
            K.CenterCrop(self.patch_size),
            NormalizeMeanStd(mean=mean, std=std),
            data_keys=["image"],
        )
        
    def setup(self, stage: str = None) -> None:
        """
        Override to define train/val/test/predict datasets for the given stage.
        """
        if stage in ["fit"]:
            self.train_dataset = self.dataset_class(split="train", **self.kwargs)
        if stage in ['fit', 'validate']:
            self.val_dataset = self.dataset_class(split="val", **self.kwargs)
        if stage in ['test']:
            self.test_dataset = self.dataset_class(split="test", **self.kwargs)
        if stage in ["predict"]: 
            self.predict_dataset = self.dataset_class(split="test", **self.kwargs)
    
    def on_after_batch_transfer(
        self, batch: dict[str, Tensor], dataloader_idx: int
    ) -> dict[str, Tensor]:
        """Apply batch augmentations after transferring batch to device.

        Args:
            batch: A batch of data that needs to be augmented.
            dataloader_idx: The index of the dataloader to which the batch belongs.

        Returns:
            Augmented batch of data.
        """
        if self.trainer:
            if self.trainer.training:
                aug = self.train_aug
            elif self.trainer.validating or self.trainer.sanity_checking:
                aug = self.val_aug
            elif self.trainer.testing or self.trainer.predicting:
                aug = self.test_aug
            else:
                raise NotImplementedError("Unknown trainer state")

            batch["image"] = batch["image"].float()
            batch = aug(batch)
            batch["image"] = batch["image"].to(batch["label"].device)


        return batch