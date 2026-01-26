import os
from typing import Callable, Optional, List, Union
from pathlib import Path
import torch
from torch import Tensor
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from torchgeo.datasets.geo import NonGeoDataset

class EnMAPBNETDDataset(NonGeoDataset):
    """EnMAP-BNETD dataset for land cover monitoring with hyperspectral imagery.
    
    This dataset pairs EnMAP hyperspectral imagery with a land cover map from
    the BNETD (Bureau National d'Études Techniques et de Développement) database
    over Ivory Coast. 

    Expects a directory layout:
      root/
        enmap/               # hyperspectral images (C,H,W)
        bnetd/               # label rasters (1,H,W) with CDL class IDs
      splits/
        enmap_bnetd/
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
    S2L2A_INDICES = [6, 16, 30, 48, 54, 59, 65, 71, 75, 90, 131, 172]
    RGB_INDICES = [3, 2, 1]
    # Relative directories (with respect to the root)
    IMAGE_ROOT = "enmap"         # e.g., directory containing images
    MASK_ROOT = "bnetd"    # e.g., directory containing masks
    
    def __init__(
        self,
        root: str = "./data/enmap_bnetd",
        sensor: str = "enmap",
        split: str = "train",
        classes: Optional[List[int]] = None,
        transforms: Optional[Callable[[dict[str, Tensor]], dict[str, Tensor]]] = None,
        num_bands: int = 12,
        band_selection: str = "naive",
        indices: Optional[list[int]] = None,
        srf_weight_matrix: Union[str, Path] = None,
        raw_mask: bool = False,
    ) -> None:
        """Initialize the EnMAP-BNETD dataset.
        
        Args:
            root: Root directory where dataset is stored
            sensor: Sensor name (default: "enmap")
            split: Dataset split ("train", "val", or "test")
            classes: List of forest class codes (must include 0 for background)
            transforms: Optional transforms to apply to samples
            num_bands: Number of spectral bands.
            band_selection (str): Method of mapping the hyperspectral bands into a lower space ("naive" or "srf_grouping").
            indices (list[int], optional): which indices to select for naive band selection.
            srf_weight_matrix (str | Path): Path to the weight matrix used in srf grouping method.
            raw_mask: If True, don't remap mask classes
            
        Raises:
            ValueError: If split is invalid, classes is None, 0 not in classes,
                or split file is not found
        """
        super().__init__()
        if split not in self.VALID_SPLITS:
            raise ValueError(f"Split '{split}' not one of {self.VALID_SPLITS}.")
        if classes is None:
            raise ValueError("Please provide a list of classes. All other classes will be mapped to background.")
        if 0 not in classes:
            raise ValueError("The provided classes must include the background code 0.")
        
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
        self.root = Path(root)
        self.transforms = transforms
        self.raw_mask = raw_mask
        self.sensor = sensor
        self.num_bands = num_bands

        # Store the provided classes.
        self.classes = classes
        # Foreground classes are all classes except 0, in the order provided.
        self.foreground_classes = [c for c in classes if c != 0]
        # The background/ignore index is set as the last index.
        self.ignore_index = len(self.foreground_classes)
        # Build mapping for foreground classes: each foreground class code is mapped to a new ordinal (0-based)
        self.mapping = {code: idx for idx, code in enumerate(self.foreground_classes)}
        
        # Build colormap for visualization: total classes = len(foreground_classes) + 1 (background)
        num_vis_classes = len(self.foreground_classes) + 1
        cmap = cm.get_cmap("tab20", num_vis_classes)
        self.ordinal_cmap = torch.zeros((num_vis_classes, 4), dtype=torch.uint8)
        for i in range(num_vis_classes):
            color = cmap(i)  # (r, g, b, a) in [0, 1]
            self.ordinal_cmap[i] = torch.tensor([int(255 * c) for c in color])
        
        # Read split file containing sample identifiers
        self.split_file = self.root.parent / "splits" / "enmap_bnetd" / f"{self.split}.txt"
        if os.path.exists(self.split_file):
            self.sample_collection = self.read_split_file()
        else:
            raise ValueError(f"Split file {self.split_file} not found.")

    def read_split_file(self) -> List:
        """Read the split file (a text file with one sample identifier per line) and return a list of (image_path, mask_path) tuples."""
        with open(self.split_file, "r") as f:
            sample_ids = [x.strip() for x in f.readlines()]
        sample_collection = [
            (
                self.root / self.IMAGE_ROOT / sample_id,
                self.root / self.MASK_ROOT / sample_id
            )
            for sample_id in sample_ids
        ]
        return sample_collection

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return a sample given its index."""
        img_path, mask_path = self.sample_collection[index]
        sample = {
            "image": self._load_image(img_path),
            "mask": self._load_mask(mask_path)
        }
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.sample_collection)

    def _load_image(self, path: str) -> Tensor:
        """Load the TreeMap image using rasterio."""
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
        """Load the TreeMap mask using rasterio, and remap classes if needed."""
        with rasterio.open(path) as src:
            mask = torch.from_numpy(src.read(1)).long()  # shape: (1, H, W)
        if self.raw_mask:
            return mask
        return self._remap_mask(mask)

    def _remap_mask(self, mask: Tensor) -> Tensor:
        """
        Remap the raw mask to ordinal labels.
        All pixels in the foreground classes (as provided) are remapped to [0, 1, 2, ...],
        while any other pixel (including original background, i.e. 0) is set to self.ignore_index.
        """
        mask = mask.squeeze(0)  # shape: (H, W)
        new_mask = torch.full_like(mask, fill_value=self.ignore_index)
        for code, ordinal in self.mapping.items():
            new_mask[mask == code] = ordinal
        return new_mask.unsqueeze(0)

    def plot(
        self,
        sample: dict[str, Tensor],
        show_titles: bool = True,
        suptitle: Optional[str] = None,
    ) -> plt.Figure:
        """Plot the sample image and mask."""
        ncols = 2
        image = sample["image"][self.RGB_INDICES].numpy()
        image = image.transpose(1, 2, 0)
        image = (image - image.min()) / (image.max() - image.min())

        mask = sample["mask"].squeeze(0).numpy()

        showing_predictions = "prediction" in sample
        if showing_predictions:
            pred = sample["prediction"].squeeze(0).numpy()
            ncols = 3

        fig, ax = plt.subplots(ncols=ncols, figsize=(4 * ncols, 4))
        ax[0].imshow(image)
        ax[0].axis("off")
        ax[1].imshow(self.ordinal_cmap.numpy()[mask], interpolation="none")
        ax[1].axis("off")
        if show_titles:
            ax[0].set_title("Image")
            ax[1].set_title("Mask")
        if showing_predictions:
            ax[2].imshow(self.ordinal_cmap.numpy()[pred], interpolation="none")
            if show_titles:
                ax[2].set_title("Prediction")
        if suptitle is not None:
            plt.suptitle(suptitle)
        return fig