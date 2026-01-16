from typing import Any, Union, Optional
import kornia.augmentation as K
from kornia.constants import DataKey, Resample
from torchgeo.samplers.utils import _to_tuple
from torchgeo.datamodules.geo import NonGeoDataModule
from torch import Tensor
from pathlib import Path
import torch
import numpy as np
from src.transforms.normalize import NormalizeMeanStd  
from torchgeo.datamodules.utils import MisconfigurationException
from src.datasets.enmap_bnetd import EnMAPBNETDDataset


class EnMAPBNETDDataModule(NonGeoDataModule):
    """
    LightningDataModule for the EnMAP-BNETD dataset.
    """

    def __init__(
        self,
        batch_size: int = 64,
        patch_size: Union[int, tuple[int, int]] = 128,
        num_workers: int = 0,
        stats_path: str = "data/statistics/enmap",
        band_selection: str = "srf_grouping",
        indices: Optional[list[int]] = None,
        srf_weight_file: Optional[str] = "SRF_S2L2A_EnMAP_W.npy",
        **kwargs: Any,
    ) -> None:
        """
        Args:
            batch_size: Size of each mini-batch.
            patch_size: Size of each patch (int or tuple of int).
            num_workers: Number of workers for data loading.
            stats_path: Path to the directory containing normalization statistics (mu.npy and sigma.npy).
            **kwargs: Additional keyword arguments passed to EnMAPEurocropsDataset.
                     (e.g. you must pass a list for `classes`.)
        """
        self.patch_size = _to_tuple(patch_size)
        self.band_selection = band_selection
        self.indices = indices if indices is not None else [6, 16, 30, 48, 54, 59, 65, 71, 75, 90, 131, 172]
        self.srf_weight_path = Path(stats_path, srf_weight_file)
        
        kwargs["band_selection"] = band_selection
        kwargs["indices"] = self.indices
        kwargs["srf_weight_matrix"] = self.srf_weight_path
        
        super().__init__(EnMAPBNETDDataset, batch_size, num_workers, **kwargs)
        
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
            
        self.train_aug = K.AugmentationSequential(
            K.Resize(_to_tuple(self.patch_size)),
            K.RandomResizedCrop(_to_tuple(self.patch_size), scale=(0.4, 1.0)),
            K.RandomVerticalFlip(p=0.5),
            K.RandomHorizontalFlip(p=0.5),
            data_keys=None, #["image", "mask"],
            extra_args={
                DataKey.MASK: {"resample": Resample.NEAREST, "align_corners": None}
            },
        )
        self.val_aug = K.AugmentationSequential(
            K.Resize(_to_tuple(self.patch_size)),
            K.CenterCrop(self.patch_size),
            data_keys=None, #["image", "mask"],
        )
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
        """
        Apply batch augmentations after the batch is moved to the device.
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
                raise NotImplementedError("Unknown trainer mode.")
            batch["image"] = batch["image"].float()
            batch = aug(batch)
            batch["image"] = self.norm_aug(batch["image"])
            batch["image"] = batch["image"].to(batch["mask"].device)
        return batch