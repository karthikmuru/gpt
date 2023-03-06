import pytorch_lightning as pl
from torch.utils.data import DataLoader
import argparse

BLOCK_SIZE = 256
BATCH_SIZE = 64
NUM_WORKERS = 20
TRAIN_VAL_SPLIT = 0.9

class BaseDataModule(pl.LightningDataModule):

    def __init__(self, args: argparse.Namespace = None):

        self.args = vars(args) if args is not None else {}
        self.batch_size = self.args.get("batch_size", BATCH_SIZE)
        self.num_workers = self.args.get("num_workers", NUM_WORKERS)
        self.train_val_split = self.args.get("train_val_split", TRAIN_VAL_SPLIT)
        self.block_size = self.args.get("block_size", BLOCK_SIZE)
        self.data_dir = self.args["data_dir"]
        self.on_gpu = isinstance(self.args.get("gpus", None), (str, int))

        self.data_train = None
        self.data_val = None
    
    @staticmethod
    def add_to_argparse(parser):
        
        parser.add_argument(
        "--block_size", type=int, default=BLOCK_SIZE, help="Number of Blocks of Multi Head Attention."
        )
        parser.add_argument(
        "--batch_size", type=int, default=BATCH_SIZE, help="Number of examples to operate on per forward step."
        )
        parser.add_argument(
        "--num_workers", type=int, default=NUM_WORKERS, help="Number of additional processes to load data."
        )
        parser.add_argument(
        "--train_val_split", type=int, default=TRAIN_VAL_SPLIT, help="Split percentage between train and val set."
        )
        parser.add_argument(
        "--data_dir", type=str, help="Path to the data folder.", required=True
        )
        return parser
    
    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.on_gpu,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.on_gpu,
        )