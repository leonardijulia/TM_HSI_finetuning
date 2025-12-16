"""
This module contains a custom terratorch dataset for the hyperview-2 challenge dataset for patch wise regression.
"""

import glob
import torch
from pathlib import Path
import numpy as np
import pandas as pd
from torchgeo.datasets import NonGeoDataset
import albumentations as A
from torchgeo.datamodules.utils import MisconfigurationException
from src.transforms.normalize import NormalizeMeanStd

class Hyperview2NonGeo(NonGeoDataset):
    """A custom dataset for the Hyperview-2 challenge dataset for patch wise regression.
    https://ai4eo.eu/portfolio/easi-workshop-hyperview2/. This dataset comprises an input of 
    three patches of the same region (2 hyperspectral - airborne and satellite - and 1 multispectral).
    The patches have: different GSDs, varying sizes, and varying spectral resolutions.

    Args:
        root (str): Path to the root directory where the dataset is stored.
        split (str): The dataset split, one of "train", "val", or "test".
        transform (callable, optional): A function/transform that takes in an
            sample and returns a transformed version. Defaults to None.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it. Defaults to None."""
            
   
    BAND_SETS = {
        "hsi_air": {
                    "hls": {
                        "bands": ("BLUE", "GREEN", "RED", "NIR", "SWIR1", "SWIR2"),
                        "indices": [24, 46, 79, 141, 292, 400]
                    },
                    "s2l2a": {
                        "bands": ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"),
                        "indices": [9, 24, 46, 79, 91, 102, 115, 131, 141, 161, 292, 400]
                    },
                    "rgb": {
                        "bands": ("RED", "GREEN", "BLUE"),
                        "indices": [79, 46, 24]
                    },
        },
        "hsi_sat":{
                    "hls": {
                        "bands": ("BLUE", "GREEN", "RED", "NIR", "SWIR1", "SWIR2"),
                        "indices": [12, 21, 32, 52, 124, 189]
                    },
                    "s2l2a": {
                        "bands": ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"),
                        "indices": [5, 12, 21, 32, 36, 40, 44, 49, 52, 60, 124, 189]
                    },
                    "rgb": {
                        "bands": ("RED", "GREEN", "BLUE"),
                        "indices": [32, 21, 12]
                    },
        },
        "msi_sat":{
                    "hls": {
                        "bands": ("BLUE", "GREEN", "RED", "NIR", "SWIR1", "SWIR2"),
                        "indices": [1, 2, 3, 8, 10, 11]
                    },
                    "s2l2a": {
                        "bands": ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"),
                        "indices": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
                    },
                    "rgb": {
                        "bands": ("RED", "GREEN", "BLUE"),
                        "indices": [3, 2, 1]
                    },
        }
    }
    
    sensors = {
        "hsi_air": "hsi_airborne",
        "hsi_sat": "hsi_satellite", 
        "msi_sat": "msi_satellite"
    }
    
    rgb_for_vis = {"s2l2a": [3, 2, 1],
                   "hls": [2, 1, 0]}
    
    gt_file = "train_gt.csv"
    soil_params = ["B", "Cu", "Zn", "Fe", "S", "Mn"]
    
    def __init__(
            self,
            data_root: str,
            split: str = "train",
            gt_file: str = gt_file,
            stats_path: Path = None,
            target_mean: float | None = None,
            target_std: float | None = None,
            bands: str = "s2l2a",
            subset_idx: list[int] |None = None,
            sensor: str = "hsi_air",
            transform: A.Compose | None = None,):
        
        super().__init__()
        
        assert sensor in self.sensors.keys(), f"Invalid sensor '{sensor}'. Must be one of {list(self.sensors.keys())}"
        
        assert bands in self.BAND_SETS[sensor].keys(), f"Invalid band set '{bands}'. Must be one of {list(self.BAND_SETS[sensor].keys())}"
            
        try:
            mean = torch.tensor(np.load(f"{stats_path}/mu_{sensor}.npy"))
            std = torch.tensor(np.load(f"{stats_path}/sigma_{sensor}.npy"))
        except FileNotFoundError:
            raise MisconfigurationException("Missing statistics! Ensure mu.npy and sigma.npy are available.")
        
        self.split = split
        self.data_root = Path(data_root)
        self.test_root = self.data_root / "test"
        csv_file = self.data_root / gt_file
        df = pd.read_csv(csv_file)
        self.bands = self.BAND_SETS[sensor][bands]["bands"]
        self.band_indices = self.BAND_SETS[sensor][bands]["indices"]
        self.normalizer = NormalizeMeanStd(mean=mean, std=std, indices=self.band_indices)
        self.subset_idx = subset_idx
        self.samples = []
        
        if split in ["train", "val"]:
            for _, row in df.iterrows():
                patch_id = f"{int(row['sample_index']):04d}"
                patch_path = self.data_root / "train" / self.sensors[sensor] / f"{patch_id}.npz"
                
                # target vector
                label = row[self.soil_params].values.astype(np.float32)
                self.samples.append({
                    "patch_path": patch_path,
                    "label": label
                    })
                
            if subset_idx is not None:
                self.samples = [self.samples[i] for i in subset_idx]
                
        elif split == "test":
            modality = "hsi_satellite" if "hsi" in sensor else "msi_satellite"
            test_files = sorted(glob.glob(str(self.test_root / modality / "*.npz")), key=lambda x: int(Path(x).stem.split("_")[-1]))
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
            data_np = data.permute(1, 2, 0).numpy()
            mask_np = mask.permute(1, 2, 0).numpy() # Shape (H, W, 1)

            # Albumentations will use Nearest Neighbor for 'mask' and Linear for 'image'
            transformed = self.transform(image=data_np, mask=mask_np)
            
            data_np = transformed["image"]
            mask_np = transformed["mask"]

            # Back to [C, H, W]
            data = torch.from_numpy(data_np).permute(2, 0, 1).float()
            mask = torch.from_numpy(mask_np).permute(2, 0, 1).float()
        
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
    
        