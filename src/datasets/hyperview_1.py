"""
This module contains a custom terratorch dataset for the hyperview-1 challenge dataset for patch wise regression.
"""

import os
import glob
import torch
from torch import Tensor
from typing import Optional
from pathlib import Path
import numpy as np
import pandas as pd
from torchgeo.datasets import NonGeoDataset
import matplotlib.pyplot as plt
import albumentations as A
from torchgeo.datamodules.utils import MisconfigurationException
from src.transforms.normalize import NormalizeMeanStd

class Hyperview1NonGeo(NonGeoDataset):
    """A custom dataset for the Hyperview-1 challenge dataset for patch wise regression.
    https://platform.ai4eo.eu/seeing-beyond-the-visible-permanent/data

    Args:
        root (str): Path to the root directory where the dataset is stored.
        split (str): The dataset split, one of "train", "val", or "test".
        transform (callable, optional): A function/transform that takes in an
            sample and returns a transformed version. Defaults to None.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it. Defaults to None."""
            
   
    BAND_SETS = {
        "hls": {
            "bands": ("BLUE", "GREEN", "RED", "NIR"),
            "indices": [7, 32, 61, 117]
        },
        "s2l2a": {
            "bands": ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"),
            "indices": [0, 9, 30, 63, 76, 87, 101, 116, 126, 149, 149, 149]
        },
        "s2l2a_nored": {
            "bands": ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09"),
            "indices": [0, 9, 30, 63, 76, 87, 101, 116, 126, 149]
        },
        "rgb": {
            "bands": ("RED", "GREEN", "BLUE"),
            "indices": [61, 32, 7]
        },
    }
    
    rgb_for_vis = {"s2l2a": [3, 2, 1],
                   "hls": [2, 1, 0]}
    
    gt_file = 'train_gt.csv'
    soil_params = ["P", "K", "Mg", "pH"]
    
    def __init__(
            self,
            data_root: str,
            split: str = "train",
            gt_file: str = gt_file,
            stats_path: Path = None,
            target_mean: list[float] | None = None,
            target_std: list[float] | None = None,
            bands: str = 's2l2a',
            subset_idx: list[int] |None = None,
            transform: A.Compose | None = None,):
        
        super().__init__()
        
        assert bands in self.BAND_SETS.keys(), \
            f"Invalid band set '{bands}'. Must be one of {list(self.BAND_SETS.keys())}"
            
        try:
            mean = torch.tensor(np.load(f"{stats_path}/mu.npy"))
            std = torch.tensor(np.load(f"{stats_path}/sigma.npy"))
        except FileNotFoundError:
            raise MisconfigurationException("Missing statistics! Ensure mu.npy and sigma.npy are available.")

        self.normalizer = NormalizeMeanStd(mean=mean, std=std, indices="hyperview_1_nored")
        self.split = split
        self.data_root = Path(data_root)
        self.test_root = self.data_root / "test"
        self.subset_idx = subset_idx
        self.target_mean = target_mean
        self.target_std = target_std
        csv_file = self.data_root / gt_file
        df = pd.read_csv(csv_file)
        self.bands = self.BAND_SETS[bands]["bands"]
        self.band_indices = self.BAND_SETS[bands]["indices"]

        self.samples = []
        
        if split in ["train", "val"]:
            for _, row in df.iterrows():
                patch_id = int(row["sample_index"])
                patch_path = self.data_root / "train" / f"{patch_id}.npz"
                
                # target vector
                label = row[self.soil_params].values.astype(np.float32)
                self.samples.append({
                    "patch_path": patch_path,
                    "label": label
                    })
            
            if subset_idx is not None:
                self.samples = [self.samples[i] for i in subset_idx] 
        
        elif split == "test":
            test_files = sorted(glob.glob(str(test_root / "*.npz")), key=lambda x: int(Path(x).stem.split("_")[-1]))
            for file in test_files:
                self.samples.append({
                    "patch_path": Path(file)
                })
              
        self.transform = transform
              
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index: int):
        sample = self.samples[index]
        data, mask = self._load_file(sample["patch_path"])
        
        if self.transform:
            # Albumentations expects [H, W, C]
            data_np = data.permute(1, 2, 0).numpy()
            mask_np = mask.permute(1, 2, 0).numpy() # Shape (H, W, 1)

            # Albumentations will use Nearest Neighbor for 'mask' and Linear for 'image'
            transformed = self.transform(image=data_np, mask=mask_np)
            
            data_np = transformed["image"]
            mask_np = transformed["mask"]

            # Back to [C, H, W]
            data = torch.from_numpy(data_np).permute(2, 0, 1).float()
            mask = torch.from_numpy(mask_np).permute(2, 0, 1).float()
            
        # 3. concatenate the image with mask
        output_tensor = torch.cat([data, mask], dim=0)
            
        if self.split in ["train", "val"]:
            label = sample["label"]
            if self.target_mean is not None and self.target_std is not None:
                self.target_mean = np.array(self.target_mean, dtype=np.float32)
                self.target_std = np.array(self.target_std, dtype=np.float32)
                label = (label - self.target_mean) / (self.target_std)
            out = {"image": output_tensor, "label": torch.tensor(label, dtype=torch.float32)}
        else:
            out = {"image": output_tensor}
              
        return out
    
    def _load_file(self, path: str):
        npz = np.load(path)
        data = np.ma.MaskedArray(**npz)
        
        mask = (~data.mask[0]).astype(np.float32)  # all mask bands are identical
        data_selected = data[self.band_indices, :, :].filled(0).astype(np.float32)  # (C, H, W)
        
        data_tensor = torch.from_numpy(data_selected)
        norm_data = self.normalizer(data_tensor[None, :, :, :]).squeeze(0)  # (bands, H, W)
        
        mask_tensor = torch.from_numpy(mask[None, :, :])  # (1, H, W)
        norm_data = norm_data * mask_tensor  # apply mask
        
        return norm_data, mask_tensor  # (C, H, W)
    
    # def plot(self, 
    #          sample: dict[str, Tensor], 
    #          show_titles: bool = True,
    #          suptitle: Optional[str] = None,
    # ) -> plt.Figure:
    #     """Return a matplotlib figure showing the image, target, and prediction."""
    #     image = sample["image"][self.rgb_for_vis['s2l2a']].numpy()
    #     image = image.transpose(1,2,0)
    #     image = (image - image.min()) / (image.max() - image.min())
        
    #     target = sample["label"].numpy()
    #     pred = sample["prediction"].numpy()
        
    #     text = ""
    #     if target is not None:
    #         text += f"Target: \n{target}\n"
    #     if pred is not None:
    #         text += f"Prediction: \n{pred}\n"
        
    #     fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    #     ax[0].imshow(image)
    #     ax[0].axis("off")
    #     ax[1].text(0.1, 0.5, text, fontsize=11)
    #     ax[1].axis("off")
        
    #     if show_titles:
    #         ax[0].set_title("Input Image") 
            
    #     if suptitle is not None:
    #         plt.suptitle(suptitle) 
            
    #     return fig       