import torch
from pytorch_lightning.cli import LightningCLI
import Models
import Data_modules
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor,ModelCheckpoint

def cli_main():
    
    cli = LightningCLI(trainer_defaults={'logger': WandbLogger()},save_config_kwargs={"overwrite": True})
    # note: don't call fit!!


if __name__ == "__main__":
    cli_main()
    # note: it is good practice to implement the CLI in a function and call it in the main if block