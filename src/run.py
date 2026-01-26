"""
Unified training/testing/prediction script.

Usage:
    python src/run.py mode=train experiment=hyperview_1
    python src/run.py mode=test experiment=hyperview_1 ckpt_path=outputs/...
    python src/run.py mode=predict experiment=hyperview_1 ckpt_path=outputs/...
"""

import logging
from pathlib import Path
from datetime import datetime
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from lightning import Trainer, seed_everything
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from lightning.pytorch.callbacks import ModelCheckpoint

log = logging.getLogger(__name__)

def setup_common(cfg: DictConfig):
    """Common setup for all modes."""
    seed_everything(cfg.seed_everything, workers=True)
    log.info("Configuration:\n" + OmegaConf.to_yaml(cfg))
    
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    log.info(f"Output directory: {output_dir}")
    
    datamodule = instantiate(cfg.data)
    model = instantiate(cfg.model)
    
    return datamodule, model, output_dir

def setup_trainer(cfg: DictConfig) -> Trainer:
    """Setup Lightning trainer."""
    
    callbacks = []
    if "callbacks" in cfg.trainer:
        for callback_cfg in cfg.trainer.callbacks:
            callbacks.append(instantiate(callback_cfg))
    
    logger = None
        if "logger" in cfg.trainer:
        # Generate smart run_name if not set
        if not cfg.trainer.logger.init_args.get("run_name"):
            band_sel = cfg.data.init_args.get("band_selection", "default")
            run_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{band_sel}_seed{cfg.seed_everything}"
            cfg.trainer.logger.init_args.run_name = run_name
        logger = instantiate(cfg.trainer.logger)
    
    trainer_cfg = OmegaConf.to_container(cfg.trainer, resolve=True)
    trainer_cfg.pop("callbacks", None)
    trainer_cfg.pop("logger", None)
    
    return Trainer(**trainer_cfg, callbacks=callbacks, logger=logger)

@hydra.main(version_base=None, config_path="../configs", config_name="config",)
def main(cfg: DictConfig) -> None:
    """Main training/testing/predicting function."""
    
    mode = cfg.get("mode", "train")
    
    if mode == "train":
        log.info("Mode: TRAINING")
        datamodule, model, output_dir = setup_common(cfg)
        trainer = setup_trainer(cfg)
        trainer.fit(model, datamodule=datamodule)
        
    elif mode == "test":
        log.info("Mode: TESTING")
        datamodule, model, output_dir = setup_common(cfg)
        trainer = setup_trainer(cfg)
        
        ckpt_path = cfg.get("ckpt_path")
        if not ckpt_path:
            raise ValueError("ckpt_path required for test mode")
        
        trainer.test(model, datamodule=datamodule, ckpt_path=ckpt_path)
        
    elif mode == "predict":
        log.info("Mode: PREDICTION")
        datamodule, model, output_dir = setup_common(cfg)
        trainer = setup_trainer(cfg)
        
        ckpt_path = cfg.get("ckpt_path")
        if not ckpt_path:
            raise ValueError("ckpt_path required for predict mode")
        
        predictions = trainer.predict(model, datamodule=datamodule, ckpt_path=ckpt_path)
        
        # Save predictions
        import pickle
        pred_file = output_dir / "predictions.pkl"
        with open(pred_file, "wb") as f:
            pickle.dump(predictions, f)
        log.info(f"Predictions saved to {pred_file}")
    
    else:
        raise ValueError(f"Unknown mode: {mode}. Choose: train, test, predict")

if __name__ == "__main__":
    main()