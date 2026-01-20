"""
This module contains a custom terratorch dataset for the hyperview-1 challenge dataset for patch wise regression.
"""
import glob
import torch
import numpy as np
import pandas as pd
import albumentations as A
from pathlib import Path
from typing import Optional, Union
from torchgeo.datasets import NonGeoDataset
import matplotlib.pyplot as plt
from torchgeo.datamodules.utils import MisconfigurationException
from src.transforms.normalize import NormalizeMeanStd

class Hyperview1NonGeo(NonGeoDataset):
    """A custom dataset for the Hyperview-1 challenge dataset for patch wise regression.
    https://platform.ai4eo.eu/seeing-beyond-the-visible-permanent/data
    """
   
    BAND_SETS = {
        "hls": { "indices": [7, 32, 61, 117] },
        "s2l2a": { "indices": [0, 9, 30, 63, 76, 87, 101, 116, 126, 149, 149, 149]},
        "s2l2a_nored": { "indices": [0, 9, 30, 63, 76, 87, 101, 116, 126, 149]},
        "rgb": {"indices": [61, 32, 7]},
    }
    
    RGB_VIS = {"s2l2a": [3, 2, 1],
                   "hls": [2, 1, 0]}

    SOIL_PARAMS = ["P", "K", "Mg", "pH"]
    VALID_SPLITS = ["train", "val", "test"]
    VALID_BAND_SELECTION = ["naive", "srf_grouping"]
    
    def __init__(
            self,
            root: str = "./data/hyperview_1",
            split: str = "train",
            gt_file: str = "train_gt.csv",
            stats_path: Optional[Union[str, Path]] = None,
            target_mean: Optional[list[float]] = None,
            target_std: Optional[list[float]] = None,
            band_selection: str = "naive",
            bands: str = "s2l2",
            srf_weight_matrix: Optional[Union[str, Path]] = None,
            subset_idx: Optional[list[int]] = None,
            transform: Optional[A.Compose] = None,
        ):
        super().__init__()
        
        """Initializes the Hyperview-1 dataset.

        Args:
            root (str, optional): Root directory where the dataset is stored. Defaults to "./data/hyperview_1".
            split (str, optional): Dataset split. Defaults to "train".
            gt_file (str, optional): Name of the csv file containing the ground truth. Defaults to gt_file.
            stats_path (Path, optional): Path to the statistics used for data normalization. Defaults to None.
            target_mean (list[float] | None, optional): Mean of the targets. Defaults to None.
            target_std (list[float] | None, optional): Std of the targets. Defaults to None.
            band_selection (str): Method of mapping the hyperspectral bands into a lower space ("naive" or "srf_grouping).
            bands (str, optional): Name of the sensor, which bands will be mapped with naive band extraction. Defaults to 's2l2a'.
            srf_weight_matrix (str | Path): Path to the weight matrix used in srf grouping method.
            subset_idx (list[int] | None, optional): Indexes used for the split of the data into train/val. 
                        Passed through the dataloader. Defaults to None.
            transform (A.Compose | None, optional): Dataset transformations to be applied. Defaults to None.

        Raises:
            MisconfigurationException: _description_
        """
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.band_selection = band_selection
        self.bands_name = bands
        
        if split not in self.VALID_SPLITS:
            raise ValueError(f"Split '{split}' must be one of {self.VALID_SPLITS}.")
        if bands not in self.BAND_SETS:
            raise ValueError(f"Invalid band set '{bands}'. Options: {list(self.BAND_SETS.keys())}")
        if band_selection not in self.VALID_BAND_SELECTION:
            raise ValueError(f"Band selection must be one of {self.VALID_BAND_SELECTION}")
            
        if stats_path is None:
             raise MisconfigurationException("stats_path must be provided.")
        
        stats_path = Path(stats_path)
        try:
            raw_mean = torch.tensor(np.load(stats_path / "mu.npy"))
            raw_std = torch.tensor(np.load(stats_path / "sigma.npy"))
        except FileNotFoundError:
            raise MisconfigurationException(f"Missing mu.npy or sigma.npy in {stats_path}")

        self.band_indices = self.BAND_SETS[bands]["indices"]
        
        if self.band_selection == "naive":
            self.mean = raw_mean[self.band_indices]
            self.std = raw_std[self.band_indices]
            self.srf_weights = None

        elif self.band_selection == "srf_grouping":
            if srf_weight_matrix is None:
                raise ValueError("srf_grouping requires srf_weight_matrix path.")
            
            self.srf_weights = np.load(srf_weight_matrix).astype(np.float32)
            self.mean = torch.matmul(raw_mean, torch.from_numpy(self.srf_weights))
            raw_var = raw_std ** 2
            weights_sq = torch.from_numpy(self.srf_weights) ** 2
            proj_var = torch.matmul(raw_var, weights_sq)
            self.std = torch.sqrt(proj_var)

        self.normalizer = NormalizeMeanStd(mean=self.mean, std=self.std)
      
        self.target_mean = np.array(target_mean, dtype=np.float32) if target_mean else None
        self.target_std = np.array(target_std, dtype=np.float32) if target_std else None

        self.samples = self._prepare_samples(gt_file, subset_idx)
        
    def _prepare_samples(self, gt_file: str, subset_idx: Optional[List[int]]) -> List[Dict]:
        """Parses the CSV and prepares the list of samples."""
        samples = []
        
        if self.split in ["train", "val"]:
            csv_path = self.root / gt_file
            if not csv_path.exists():
                raise FileNotFoundError(f"GT file not found at {csv_path}")
                
            df = pd.read_csv(csv_path)
            
            # Filter by subset_idx if provided (e.g. from random_split in DataModule)
            if subset_idx is not None:
                df = df.iloc[subset_idx]

            for _, row in df.iterrows():
                patch_id = int(row["sample_index"])
                patch_path = self.root / "train" / f"{patch_id}.npz"
                label = row[self.SOIL_PARAMS].values.astype(np.float32)
                samples.append({"patch_path": patch_path, "label": label})
        
        elif self.split == "test":
            test_root = self.root / "test"
            # Sort naturally: 1.npz, 2.npz ... 10.npz
            test_files = sorted(test_root.glob("*.npz"), 
                                key=lambda x: int(x.stem.split("_")[-1]) if "_" in x.stem else int(x.stem))
            for file in test_files:
                samples.append({"patch_path": file})
                
        return samples
              
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
            
            data = torch.from_numpy(transformed["image"]).permute(2, 0, 1).float()
            mask = torch.from_numpy(transformed["mask"]).permute(2, 0, 1).float()
            
        data = data * mask # applying mask
        output_tensor = torch.cat([data, mask], dim=0) # additionally concatenating 

        out = {"image": output_tensor}

        if "label" in sample:
            label = sample["label"]
            # Normalize target if statistics provided
            if self.target_mean is not None and self.target_std is not None:
                label = (label - self.target_mean) / self.target_std
            out["label"] = torch.tensor(label, dtype=torch.float32)

        return out
    
    def _load_file(self, path: str):
        """Loads .npz file and handles band selection/projection."""
        with np.load(path) as npz:
            raw_data = npz['data']
            raw_mask = npz['mask'] 
            
        data = np.ma.MaskedArray(data=raw_data, mask=raw_mask)
        valid_mask = (~raw_mask[0]).astype(np.float32) # all masks are identical
        
        if self.band_selection == "naive":
            data_selected = data[self.band_indices, :, :].filled(0).astype(np.float32)  # (C, H, W)
            #data_tensor = torch.from_numpy(data_selected).float()
            
        elif self.band_selection == "srf_grouping":
            #ASSERT C?
            if isinstance(raw_data, np.ma.MaskedArray):
                raw_data = raw_data.filled(0)
            
            # Projection: Sum_over_c (Pixel_c * Weight_c_o)
            data_selected = np.einsum('chw,co->ohw', raw_data, self.srf_weights)
            
        data_tensor = torch.from_numpy(data_selected).float()
        mask_tensor = torch.from_numpy(valid_mask).unsqueeze(0).float() # (1, H, W)

        return data_tensor, mask_tensor
    
    # def plot(self, 
    #          sample: dict[str, Tensor], 
    #          show_titles: bool = True,
    #          suptitle: Optional[str] = None,
    # ) -> plt.Figure:
    #     """Return a matplotlib figure showing the image, target, and prediction."""
    #     image = sample["image"][self.RGB_VIS['s2l2a']].numpy()
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