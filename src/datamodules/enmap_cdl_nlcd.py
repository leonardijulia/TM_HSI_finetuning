from typing import Any, Union, Optional
import torch
import numpy as np
from pathlib import Path
import kornia.augmentation as K
from kornia.constants import DataKey, Resample
from torchgeo.datamodules.geo import NonGeoDataModule
from torchgeo.samplers.utils import _to_tuple
from torchgeo.datamodules.utils import MisconfigurationException
from torch import Tensor
from src.datasets.enmap_cdl_nlcd import EnMAPCDLNLCDDataset
from src.transforms.normalize import NormalizeMeanStd

class EnMAPCDLNLCDDataModule(NonGeoDataModule):
    """LightningDataModule for the EnMAP-CDL/NLCD dataset."""
    
    def __init__(
        self,
        batch_size: int = 64,
        patch_size: Union[int, tuple[int, int]] = 128,
        num_workers: int = 0,
        stats_path: str = "statistics",
        band_selection: str = "naive",
        indices: Optional[list[int]] = None,
        srf_weight_file: Optional[str] = "SRF_S2L2A_EnMAP_W.npy",
        **kwargs: Any,
    ) -> None:
        """Initialize an EnMAP-CDL/NLCD datamodule.

        Args:
            batch_size: Size of each mini-batch for training, validation, and testing.
            patch_size: Size of each image patch, either a single int or (height, width) tuple.
                All images will be resized to this size. Default: 128.
            num_workers: Number of workers for parallel data loading. Default: 0.
            stats_path: Path to the directory containing normalization statistics (mu.npy, sigma.npy,
                and srf_weights if available). Default: "data/statistics".
            band_selection: Either 'naive' band selection or 'srf_grouping'.
            **kwargs: Additional keyword arguments passed to the EnMAPCDLNLCDDataset constructor.
                
        Raises:
            MisconfigurationException: If the normalization statistics files are not found
                at the specified stats_path.
        """
        self.patch_size = _to_tuple(patch_size)
        self.band_selection = band_selection
        self.indices = indices if indices is not None else [6, 16, 30, 48, 54, 59, 65, 71, 75, 90, 131, 172]
        self.srf_weight_path = Path(stats_path, srf_weight_file)
        self.aug = None
        
        kwargs["band_selection"] = band_selection
        kwargs["indices"] = self.indices
        kwargs["srf_weight_matrix"] = self.srf_weight_path
        
        super().__init__(EnMAPCDLNLCDDataset, batch_size, num_workers, **kwargs)
        
        try:
            raw_mean = torch.tensor(np.load(f"{stats_path}/mu.npy"))
            raw_std = torch.tensor(np.load(f"{stats_path}/sigma.npy"))
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
            
        # Training augmentations: resize, random crop, flips, and normalization
        self.train_aug = K.AugmentationSequential(
            K.Resize(_to_tuple(self.patch_size)),
            K.RandomResizedCrop(_to_tuple(self.patch_size), scale=(0.2, 1.0)),
            K.RandomVerticalFlip(p=0.5),
            K.RandomHorizontalFlip(p=0.5),
            data_keys=None, #["image", "mask"],
            extra_args={
                DataKey.MASK: {"resample": Resample.NEAREST, "align_corners": None}
            },
        )
        
        # Validation augmentations: resize, center crop, and normalization
        self.val_aug = K.AugmentationSequential(
            K.Resize(_to_tuple(self.patch_size)),
            K.CenterCrop(self.patch_size),
            data_keys=None, #["image", "mask"],
        )
        
        # Test augmentations: same as validation
        self.test_aug = K.AugmentationSequential(
            K.Resize(_to_tuple(self.patch_size)),
            K.CenterCrop(self.patch_size),
            data_keys=None, #["image", "mask"],
        )
        
        self.norm_aug = K.AugmentationSequential(
            NormalizeMeanStd(mean=mean, std=std),
            data_keys=["image"])
    
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
        """Apply augmentations to the batch after it is transferred to the device.

        This method is called automatically by PyTorch Lightning after the batch
        is transferred to the device (GPU/CPU). It applies the appropriate augmentations
        based on the current trainer state (training/validation/testing).
        
        
        Args:
            batch: A dictionary batch of data containing 'image' and 'mask' tensors.
            dataloader_idx: The index of the dataloader to which the batch belongs.

        Returns:
            The augmented batch with normalized images and masks.
            
        Raises:
            NotImplementedError: If the trainer mode cannot be determined.
        """
        if self.trainer:
            if self.trainer.training:
                aug = self.train_aug 
            elif self.trainer.validating or self.trainer.sanity_checking:
                aug = self.val_aug
            elif self.trainer.testing:
                aug = self.test_aug 
            elif self.trainer.predicting:
                aug = self.test_aug
            else:
                print("No trainer mode found")
                raise NotImplementedError 
            batch["image"] = batch["image"].float()
            batch = aug(batch)
            batch["image"] = self.norm_aug(batch["image"])
            batch["image"] = batch["image"].to(batch["mask"].device)

        return batch