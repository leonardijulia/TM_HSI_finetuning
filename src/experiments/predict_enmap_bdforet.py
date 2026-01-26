import os
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
from lightning.pytorch import Trainer
from terratorch.tasks import SemanticSegmentationTask
from src.datamodules.enmap_bdforet import EnMAPBDForetDataModule
from src.datasets.enmap_bdforet import EnMAPBDForetDataset

def plot_sample(image, label, num_classes, save_path, prediction=None, suptitle=None, class_names=None, show_axes=False):
    num_images = 4 if prediction is not None else 3
    fig, ax = plt.subplots(1, num_images, figsize=(12, 10), layout="compressed")
    axes_visibility = "on" if show_axes else "off"
    image = image.permute(1,2,0)
    label = label.permute(1,2,0)
    # for legend
    ax[0].axis("off")
    colors = [
    "#7c580b",  # Chesnut
    "#134606",  # Deciduous oaks
    "#95dd64",  # Evergreen oaks
    "#70cbce",  # Beech
    "#ba46c9",  # Robina
    "#c6c910",  # Douglas-fir
    "#4a8f4d",  # Hooked/Cembro pine
    "#ff9100",  # Other pine
    "#c5b28e",  # Aleppo pine
    "#520d0d",  # Corsican/black pine
    "#f80606",  # Scots pine
    "#9caa82",  # Fir/Spruce  
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
    ckpt_path = "./outputs/enmap_bdforet/checkpoints/121225_epoch19_rerun.ckpt"
    data_root = "./data/enmap_bdforet"
    out_dir = "./outputs/enmap_bdforet/inference_results"
    os.makedirs(out_dir, exist_ok=True)

    # ----------------------------------------------------------
    # 1. Load Terratorch Model

    model = SemanticSegmentationTask.load_from_checkpoint(ckpt_path)
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    # ----------------------------------------------------------
    # 2. Load DataModule → gives us test_dataloader()
    #    Must pass SAME arguments as training (batch_size, patch_size…)

    dm = EnMAPBDForetDataModule(
        root=data_root,
        batch_size=1,
        patch_size=224,
        num_workers=2,
    )
    dm.setup("test")

    # ----------------------------------------------------------
    # 3. Generate predictions using a trainer

    trainer = Trainer(accelerator="auto", strategy="auto", devices="auto", num_nodes=1, precision="16-mixed")
    predictions = trainer.predict(model, datamodule=dm)

    all_preds = torch.cat([p[0][0] for p in predictions], dim=0)
    test_loader = dm.test_dataloader()
    class_names = ["Chesnut", "Deciduous oaks", "Evergreen oaks", "Beech", "Robina", "Douglas-fir", 
                    "Hooked/Cembro pine", "Other pine", "Aleppo pine", "Corsican/Black pine", "Scots pine", "Fir/Spruce", "Background", ]

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