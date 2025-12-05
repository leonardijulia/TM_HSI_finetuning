import os
import torch
import terratorch
import optuna
import lightning.pytorch as pl
from pathlib import Path

from src.datamodules.hyperview_1 import Hyperview1NonGeoDataModule
from terratorch.tasks import ScalarRegressionTask

# --- Global Config ---
EXPERIMENT_ROOT = "/leonardo/home/userexternal/jleonard/experiments/outputs/hyperview/hpo"
DATA_ROOT = "/leonardo/home/userexternal/jleonard/experiments/data/hyperview_1/"
BACKBONE_CKPT = "/leonardo/home/userexternal/jleonard/experiments/src/ckpt/TerraMind_v1_base.pt"

def create_model(trial_params):
  """ Creates the Terratorch Task based on Optuna parameters. """
  model = ScalarRegressionTask(
    model_factory = "EncoderDecoderFactory",
    model_args = {
      "backbone": "terramind_v1_base",
      "backbone_ckpt_path": BACKBONE_CKPT,
      "backbone_modalities": ["S2L2A"],
      "backbone_img_size": 224,
      "backbone_in_chans": 13,
      "backbone_bands": {"S2L2A": ["COASTAL_AEROSOL", "BLUE", "GREEN", "RED", "RED_EDGE_1", "RED_EDGE_2", 
                                   "RED_EDGE_3", "NIR_BROAD", "NIR_NARROW", "WATER_VAPOR", "SWIR_1", "SWIR_2", "MASK"]},

      "necks": [{"name": "SelectIndices",
                 "indices": [-1]},
                {"name": "PermuteDims",
                 "new_order": [0,2,1]},
                ],

      "decoder": "IdentityDecoder",
      
      "head_dropout": trial_params['head_dropout'],
      "head_dim_list": trial_params['head_dim_list'],
      "head_linear_after_pool": trial_params['head_linear_after_pool'],
      "head_num_outputs": 4,
    },
    
    num_outputs = 4,
    loss = "mse",
    optimizer = "AdamW",
    optimizer_hparams = {"weight_decay" : trial_params["weight_decay"]},
    freeze_backbone = False,
    freeze_decoder = False,
    var_names = ["P", "K", "Mg", "pH"],
    path_to_record_metrics = os.path.join(EXPERIMENT_ROOT, f"trial_{trial_params['trial_id']}"),
    lr = trial_params['lr'],
    plot_on_val = False,
)
  return model

def objective(trial):
  head_dim_key = trial.suggest_categorical("head_dim_config", ["none", "small", "medium"])
  head_dim_map = {
        "none": None,
        "small": [256],
        "medium": [512, 256]
    }
    
  params = {
        "trial_id": trial.number,
        "lr": trial.suggest_float("lr", 5e-5, 5e-4, log=True),
        "weight_decay": trial.suggest_categorical("weight_decay", [0.01, 0.03, 0.05, 0.1]),
        "head_dropout": trial.suggest_float("dropout", 0.2, 0.4),
        "head_linear_after_pool": trial.suggest_categorical("linear_after_pool", [True, False]),
        "head_dim_list": head_dim_map[head_dim_key],
        "batch_size": trial.suggest_categorical("batch_size", [32, 64])
  }
    
  datamodule = Hyperview1NonGeoDataModule(
        batch_size=params['batch_size'],
        num_workers=2,
        data_root=DATA_ROOT
  )
    
  model = create_model(params)
    
  pruning_callback = optuna.integration.PyTorchLightningPruningCallback(trial, monitor="val/loss")
    
  checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(EXPERIMENT_ROOT, f"checkpoints/trial_{trial.number}"),
        monitor="val/loss",
        save_top_k=1,
        mode="min",
        save_weights_only=True,
        filename="{epoch:02d}-{val/loss:.4f}"
  )

  logger = pl.loggers.CSVLogger(save_dir=EXPERIMENT_ROOT, name=f"trial_{trial.number}")

  trainer = pl.Trainer(
    accelerator="auto",
    devices="auto",
    strategy="auto",
    num_nodes=1,
    precision="16-mixed",
    logger=logger,
    max_epochs=20, # Reduced epochs for HPO, verify with full run later
    enable_checkpointing=True,
    log_every_n_steps=10,
    callbacks=[
      checkpoint_callback,
      pruning_callback, 
      pl.callbacks.LearningRateMonitor(logging_interval="epoch"),
      # Safety net: stop if loss diverges
      pl.callbacks.EarlyStopping(monitor="val/loss", patience=5, mode="min") 
    ],
    default_root_dir=EXPERIMENT_ROOT,
    enable_progress_bar=False, # cleaner output in logs
  )
  
  try:
    trainer.fit(model, datamodule=datamodule)
        
    # Retrieve metric safely
    if trainer.callback_metrics.get("val/loss") is not None:
      val_loss = trainer.callback_metrics["val/loss"].item()
    else:
      val_loss = float('inf') # Penalty for failure
            
  except Exception as e:
    print(f"Trial {trial.number} failed with error: {e}")
    val_loss = float('inf')

  if trial.should_prune():
        import shutil
        ckpt_dir = os.path.join(EXPERIMENT_ROOT, f"checkpoints/trial_{trial.number}")
        if os.path.exists(ckpt_dir):
            shutil.rmtree(ckpt_dir) # Deletes the whole folder
            print(f"Trial {trial.number} pruned. Checkpoint deleted.")
            
  return val_loss

if __name__ == "__main__":
  # Ensure directory exists
  os.makedirs(EXPERIMENT_ROOT, exist_ok=True)
    
  # Storage setup: Saves the study to a local file so you can resume if it crashes
  db_path = f"sqlite:///{EXPERIMENT_ROOT}/hpo_study.db"
    
  study = optuna.create_study(
        study_name="hyperview_hpo_v1",
        direction="minimize",
        storage=db_path,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
  )
    
  print(f"Starting optimization. Results will be saved to {db_path}")
  study.optimize(objective, n_trials=50) # Run 50 trials

  print("Number of finished trials: {}".format(len(study.trials)))
  print("Best trial:")
  trial = study.best_trial

  print("  Value: {}".format(trial.value))
  print("  Params: ")
  for key, value in trial.params.items():
        print("    {}: {}".format(key, value))