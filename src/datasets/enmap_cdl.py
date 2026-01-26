# Based on: https://github.com/microsoft/torchgeo/blob/main/torchgeo/datasets/ssl4eo_benchmark.py

import os
from pathlib import Path
from typing import Callable, Optional, Union

from torchgeo.datasets.cdl import CDL
from torchgeo.datasets.geo import NonGeoDataset
import torch
from torch import Tensor

import numpy as np
import rasterio
import matplotlib.pyplot as plt
import random

class EnMAPCDLDataset(NonGeoDataset):
    """
    EnMAP hyperspectral + CDL labels for semantic segmentation.

    Expects a directory layout:
      root/
        enmap/               # hyperspectral images (C,H,W)
        cdl/                 # label rasters (1,H,W) with CDL class IDs
      splits/
        enmap_cdl/
          train.txt
          val.txt
          test.txt    

    Band handling:
      - band_selection="naive": select 12 Sentinel-2-like bands via indices.
      - band_selection="srf_grouping": load all bands and project to 12 via SRF weight matrix.
      - rgb_indices define visualization channels (assumes 12-band S2L2A order).

    Class remapping:
      - `classes` must include background 0.
      - Background is moved to the end to become the ignore class.
      - Masks are mapped to ordinal labels via `ordinal_map`; set raw_mask=True to skip remap.
    """
    
    VALID_SPLITS = ["train", "val", "test"]
    VALID_BAND_SELECTION = ["naive", "srf_grouping"]
    IMAGE_ROOT = "enmap"
    MASK_ROOT = "cdl"
    S2L2A_INDICES = [6, 16, 30, 48, 54, 59, 65, 71, 75, 90, 131, 172]
    RGB_INDICES = [3, 2, 1]

    def __init__(
        self,
        root: str = "./data/enmap_cdl",
        split: str = "train",
        classes: Optional[list[int]] = None,
        transforms: Optional[Callable[[dict[str, Tensor]], dict[str, Tensor]]] = None,
        num_bands: int = 12,
        band_selection: str = "naive",
        indices: Optional[list[int]] = None,
        srf_weight_matrix: Union[str, Path] = None,
        raw_mask: bool = False,
     ) -> None:
        
        """
        Initializes the EnMAPCDLDataset.
        
        Args:
            root: Dataset root containing enmap/, cdl/, and splits/.
            split: "train" | "val" | "test".
            classes: Class IDs to keep; others map to background. Must include 0.
            transforms: Optional sample transforms.
            num_bands: Expected band count after selection/projection (default 12).
            band_selection: "naive" or "srf_grouping".
            indices: Band indices for naive selection; defaults to S2L2A mapping.
            srf_weight_matrix: Path to SRF weight matrix (required for srf_grouping).
            raw_mask: If True, return raw CDL IDs; otherwise remap to ordinal.

        Returns:
            Samples with keys:
            - "image": float32 tensor (C,H,W)
            - "mask": long tensor (1,H,W) remapped to ordinal labels
        """
        super().__init__()
        
        assert (
            split in self.VALID_SPLITS
        ), f"Only supports one of {self.VALID_SPLITS}, but found {split}."
        self.split = split
        
        assert (
            band_selection in self.VALID_BAND_SELECTION
        ), f"Only supports one of {self.VALID_BAND_SELECTION}, but found {band_selection}"
        
        if band_selection == "srf_grouping":
            if srf_weight_matrix == None:
                raise ValueError("SRF grouping strategy requires the srf weight matrix!")                
            if not Path(srf_weight_matrix).exists():
                raise ValueError("SRF grouping matrix not found!")  
            self.srf_weight_matrix = np.load(srf_weight_matrix).astype(np.float32)
            
        self.band_selection = band_selection
        self.indices = indices if indices is not None else self.S2L2A_INDICES
        
        self.cmap = CDL.cmap
        if classes is None:
            classes = list(self.cmap.keys())

        assert 0 in classes, "Classes must include the background class: 0"

        self.root = Path(root)
        self.classes = classes
        self.transforms = transforms
        self.ordinal_map = torch.zeros(max(self.cmap.keys()) + 1, dtype=torch.long) + len(self.classes) - 1
        self.ordinal_cmap = torch.zeros((len(self.classes), 4), dtype=torch.uint8)
        self.num_bands = num_bands
        self.raw_mask = raw_mask

        # Check if split file exists, if not create it
        self.split_file = self.root.parent / "splits" / "enmap_cdl" / f"{self.split}.txt"
        if os.path.exists(self.split_file):
            self.sample_collection = self.read_split_file()
        else:
            raise ValueError(f"Split file {self.split_file} not found.")

        # First remove class 0 from the list of classes
        self.classes.remove(0)
        # Then add it back to the end of the list. This ensures that the background class is always the last class, 
        # which can be ignored during training.
        self.classes.append(0)

        # Map chosen classes to ordinal numbers, all others mapped to background class
        for v, k in enumerate(self.classes):
            self.ordinal_map[k] = v
            self.ordinal_cmap[v] = torch.tensor(self.cmap[k])
    
    def read_split_file(self) -> list:
        """Read .txt file containing train/val/test split with only image identifiers.
        """
        with open(self.split_file, "r") as f:
            sample_ids = [x.strip() for x in f.readlines()]

        # Construct the full paths for image and mask
        sample_collection = [
            (
                self.root / self.IMAGE_ROOT / sample_id,
                self.root / self.MASK_ROOT / sample_id
            )
            for sample_id in sample_ids
        ]
        
        return sample_collection

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            image and sample
        """
        img_path, mask_path = self.sample_collection[index]

        sample = {
            "image": self._load_image(img_path),
            "mask": self._load_mask(mask_path),
        }

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample


    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.sample_collection)
    
    def _load_image(self, path: str) -> Tensor:
        """Load the input image.

        Args:
            path: path to input image

        Returns:
            image
        """
        with rasterio.open(path) as src:
            if self.band_selection == "naive":
                image = src.read(self.indices)
                image = torch.from_numpy(image).float()
                
            elif self.band_selection == "srf_grouping":
                image = src.read() # (C, H, W)
                c, h, w = image.shape
                assert c == self.srf_weight_matrix.shape[0], f"Mismatch! Image has {c} bands, but weights have {self.srf_weight_matrix.shape[0]}"
                    
                # (C, H, W) -> (C, H*W) -> Transpose -> (H*W, C)
                flattened = image.reshape(c, -1).T
                grouped= np.dot(flattened, self.srf_weight_matrix)
                grouped = grouped.reshape(h, w, -1).transpose(2, 0, 1)
                image = torch.from_numpy(grouped).float()
                
        return image

    def _load_mask(self, path: str) -> Tensor:
        """Load the mask.

        Args:
            path: path to mask

        Retuns:
            mask
        """
        with rasterio.open(path) as src:
            mask = torch.from_numpy(src.read()).long()
        if self.raw_mask:
            return mask
        return self.ordinal_map[mask]

    def plot(
        self,
        sample: dict[str, Tensor],
        show_titles: bool = True,
        suptitle: Optional[str] = None,
    ) -> plt.Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional string to use as a suptitle

        Returns:
            a matplotlib Figure with the rendered sample
        """
        ncols = 2

        image = sample["image"][self.RGB_INDICES].numpy()

        image = image.transpose(1, 2, 0)
        image = (image - image.min()) / (image.max() - image.min())
        mask = sample["mask"].squeeze(0)

        showing_predictions = "prediction" in sample
        if showing_predictions:
            pred = sample["prediction"].squeeze(0)
            ncols = 3

        fig, ax = plt.subplots(ncols=ncols, figsize=(4 * ncols, 4))
        ax[0].imshow(image)
        ax[0].axis("off")
        ax[1].imshow(self.ordinal_cmap[mask], interpolation="none")
        ax[1].axis("off")
        if show_titles:
            ax[0].set_title("Image")
            ax[1].set_title("Mask")

        if showing_predictions:
            ax[2].imshow(self.ordinal_cmap[pred], interpolation="none")
            if show_titles:
                ax[2].set_title("Prediction")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig