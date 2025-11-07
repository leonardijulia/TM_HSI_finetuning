"""
This module contains a custom terratorch dataset for the hyperview-1 challenge dataset for patch wise regression.
"""

import os
import torch
from pathlib import Path
import numpy as np
import pandas as pd
from torchgeo.datasets import NonGeoDataset
import albumentations as A

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
            "bands": ("B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09"),
            "indices": [10, 31, 65, 77, 88, 102, 116, 117, 126]
        },
        "rgb": {
            "bands": ("RED", "GREEN", "BLUE"),
            "indices": [61, 32, 7]
        },
    }
    gt_file = 'train_gt.csv'
    soil_params = ["P", "K", "Mg", "pH"]
    
    def __init__(
            self,
            data_root: str,
            split: str = "train",
            gt_file: str = gt_file,
            target_mean: float | None = None,
            target_std: float | None = None,
            bands: str = 's2l2a',
            transform: A.Compose | None = None,):
        
        super().__init__()
        
        assert bands in self.BAND_SETS.keys(), \
            f"Invalid band set '{bands}'. Must be one of {list(self.BAND_SETS.keys())}"
            
        self.split = split
        self.data_root = Path(data_root)
        csv_file = self.data_root / gt_file
        df = pd.read_csv(csv_file)
        self.bands = self.BAND_SETS[bands]["bands"]
        self.band_indeces = self.BAND_SETS[bands]["indices"]

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
            
            if target_mean is not None and target_std is not None: # perhaps not - use weights only 
                self.target_mean = np.array(target_mean, dtype=np.float32)
                self.target_std = np.array(target_std, dtype=np.float32)
            else:
                # Compute mean/std from dataset if not provided
                labels = np.stack([s["label"] for s in self.samples], axis=0)
                self.target_mean = labels.mean(axis=0) # [mean("P"), mean("K"), mean("Mg"), mean("pH")]
                self.target_std = labels.std(axis=0)

                
        elif split == "test":
            for _, row in df.iterrows():
                patch_id = int(row["sample_index"])
                patch_path = self.data_root / split / f"{patch_id}.npz"
                self.samples.append({
                    "patch_path": patch_path,
                })
              
        self.transform = transform
        
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index: int):
        sample = self.samples[index]
        patch = self._load_file(sample["patch_path"])
        
        if self.transform:
            patch = self.transform(image=patch)["image"]
            
        if self.split in ["train", "val"]:
            label = sample["label"]
            label = (label - self.target_mean) / (self.target_std)
            patch = np.moveaxis(patch, -1, 0)  # (C, H, W)

            out = {"image": torch.tensor(patch, dtype=torch.float32),
                   "label": torch.tensor(label, dtype=torch.float32)}
        else:
            patch = np.moveaxis(patch, -1, 0)  # (C, H, W)
            out = {"image": torch.tensor(patch, dtype=torch.float32)}
              
        return out
    
    def _load_file(self, path: str):
        npz = np.load(path)
        data = np.ma.MaskedArray(**npz)
        data = data.filled(0)
        data = data[self.band_indeces, :, :].astype(np.float32)
        data = np.moveaxis(data, 0, -1) # (H, W, C)
        
        return data
        