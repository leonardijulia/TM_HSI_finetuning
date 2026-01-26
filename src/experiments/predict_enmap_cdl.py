import os
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
from lightning.pytorch import Trainer
from terratorch.tasks import SemanticSegmentationTask
from src.datamodules.enmap_cdl_nlcd import EnMAPCDLNLCDDataModule
from src.datasets.enmap_cdl_nlcd import EnMAPCDLNLCDDataset

def plot_sample(image, label, num_classes, save_path, prediction=None, suptitle=None, class_names=None, show_axes=False):
    num_images = 4 if prediction is not None else 3
    fig, ax = plt.subplots(1, num_images, figsize=(12, 10), layout="compressed")
    axes_visibility = "on" if show_axes else "off"
    image = image.permute(1,2,0)
    label = label.permute(1,2,0)
    # for legend
    ax[0].axis("off")
    colors = [
        "#d55e00",  # Corn - burnt orange
        "#0072b2",  # Cotton - steel blue
        "#f0e442",  # Rice - mustard yellow
        "#009e73",  # Sorghum - dark teal
        "#e69f00",  # Soybeans - amber
        "#56b4e9",  # Sunflower - sky blue
        "#cc79a7",  # Sugarcane - muted magenta
        "#f0a202",  # Tomatoes - golden orange
        "#6a3d9a",  # Grapes - muted purple
        "#8c564b",  # Citrus - warm brown
        "#1f77b4",  # Almonds - steel blue variant
        "#ff7f0e",  # Walnuts - orange variant
        "#2ca02c",  # Pistachios - forest green
        "#d62728",  # Prunes - muted red
        "#3d3b3b",  # Background -> pastel gray
    ]

    cmap = mcolors.ListedColormap(colors)
    norm = mpl.colors.Normalize(vmin=0, vmax=num_classes - 1)
    for i in range(image.shape[0]):
        standardized_band = (image[i] - image[i].min()) / (image[i].max() - image[i].min())
        image[i] = standardized_band
    ax[1].axis(axes_visibility)
    ax[1].title.set_text("Image")
    ax[1].imshow(image)

    ax[2].axis(axes_visibility)
    ax[2].title.set_text("Ground Truth Mask")
    ax[2].imshow(label, cmap=cmap, norm=norm)

    if prediction is not None:
        ax[3].axis(axes_visibility)
        ax[3].title.set_text("Predicted Mask")
        ax[3].imshow(prediction, cmap=cmap, norm=norm)

    #cmap = plt.get_cmap(cmap)
    legend_data = []
    for i, _ in enumerate(range(num_classes)):
        class_name = class_names[i] if class_names else str(i)
        data = [i, cmap(norm(i)), class_name]
        legend_data.append(data)
    handles = [Rectangle((0, 0), 1, 1, color=tuple(v for v in c)) for k, c, n in legend_data]
    labels = [n for k, c, n in legend_data]
    ax[0].legend(handles, labels, loc="center")
    if suptitle is not None:
        plt.suptitle(suptitle)
    fig.savefig(save_path)
    plt.close(fig)

def main():
    ckpt_path = "./outputs/enmap_cdl/checkpoints/epoch70_150126.ckpt"
    data_root = "./data/enmap_cdl"
    out_dir = "./outputs/enmap_cdl/inference_results"
    os.makedirs(out_dir, exist_ok=True)

    # 1. Load Terratorch Model

    model = SemanticSegmentationTask.load_from_checkpoint(ckpt_path)
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Load DataModule → gives us test_dataloader()
    #    Must pass SAME arguments as training (batch_size, patch_size…)

    dm = EnMAPCDLNLCDDataModule(
        root=data_root,
        batch_size=1,
        patch_size=224,
        num_workers=2,
        stats_path="data/statistics/enmap/",
        band_selection="srf_grouping"
    )
    dm.setup("test")

    # 3. Generate predictions using a trainer

    trainer = Trainer(accelerator="auto", strategy="auto", devices="auto", num_nodes=1, precision="16-mixed")
    predictions = trainer.predict(model, datamodule=dm)

    all_preds = torch.cat([p[0][0] for p in predictions], dim=0)
    test_loader = dm.test_dataloader()
    class_names = ["Corn", "Cotton", "Rice", "Sorghum", "Soybeans", "Sunflower", "Sugarcane", 
                  "Tomatoes", "Grapes", "Citrus", "Almonds", "Walnuts", "Pistachios", "Prunes", "Background"]

    for i, batch in enumerate(test_loader):
        img = batch["image"][0, [3,2,1], :, :]  # RGB
        label = batch["mask"][0]

        # Extract prediction
        pred = all_preds[i].cpu()  # (1,H,W) or (H,W)

        save_path = os.path.join(out_dir, f"sample_{i:04d}.png")
        plot_sample(img, label, num_classes=len(class_names), save_path=save_path,
                    prediction=pred, class_names=class_names, suptitle=f"Sample {i}")

        if i % 10 == 0:
            print(f"Saved {save_path}")

    print("Inference done!")


if __name__ == "__main__":
    main()