import os
import glob
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras.utils import Sequence
import pandas as pd

join = os.path.join


class nih_dataset_classification(Sequence):
    def __init__(self, dir='~/data/nih'):
        self.df = pd.read_csv(join(dir, 'Data_Entry_2017.csv'))
        self.data_files = self.df[['Image Index', 'Finding Labels']]
        self.files = glob.glob(dir + '/**/*.png', recursive=True, )
        print(len(self.files))
    def __len__(self,):
        pass

    def __getitem__(self, idx):
        pass

if __name__ == "__main__":
    dataset = nih_dataset_classification()