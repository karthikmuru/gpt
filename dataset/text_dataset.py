import argparse
from base_data_module import BaseDataModule
from text_data import TextData, TextFile

class TextDataset(BaseDataModule):

    def __init__(self, args: argparse.Namespace = None):
        super().__init__(self, args)

        self.text_file = TextFile(self.data_dir, self.train_val_split)
        
        self.data_train = TextData(self.text_file.tokens())
        self.data_test = TextData(self.text_file.tokens('test'))