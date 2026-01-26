import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from lightning.pytorch import Trainer
from terratorch.tasks import ScalarRegressionTask
from src.datamodules.hyperview_1 import Hyperview1NonGeoDataModule


TRAIN_MEAN = np.array([69.71876, 227.61118, 159.03355, 6.786447])
TRAIN_STD  = np.array([28.445845, 60.925327, 40.100044, 0.25799656])

CHECKPOINT_PATH = "./outputs/hyperview_1/checkpoints/last.ckpt"

if __name__ == "__main__":

    print(f"Loading model from {CHECKPOINT_PATH}...")
    model = ScalarRegressionTask.load_from_checkpoint(CHECKPOINT_PATH)
    
    dm = Hyperview1NonGeoDataModule(
        data_root="./data/hyperview_1/",
        batch_size=1, 
        num_workers=4
    )
    
    trainer = Trainer(
        accelerator="auto",
        strategy="auto", 
        devices="auto", 
        num_nodes=1, 
        precision="16-mixed", 
        logger=False,         
        enable_checkpointing=False
    )

    predictions_list = trainer.predict(model, datamodule=dm)
    
    all_preds = torch.cat([p[0] for p in predictions_list], dim=0)

    real_preds = (all_preds * TRAIN_STD) + TRAIN_MEAN

    df_results = pd.DataFrame(real_preds, columns=["P", "K", "Mg", "pH"])
    df_results.index.name = "sample_index"
    df_results.to_csv("final_submission.csv")

    print("\n✅ Success! Predictions saved to 'final_submission.csv'")
    print(f"Sample prediction (real scale): {real_preds[0]}")