import os
import glob
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras.utils import Sequence
import pandas as pd

join = os.path.join


class nih_dataset(Sequence):
    def __init__(self, dir='/home/mramados/data/nih', test=False, bounding_box=False):
        if bounding_box:
            self.df = pd.read_csv(join(dir, 'BBox_List_2017.csv'))
        else:
            self.df = pd.read_csv(join(dir, 'Data_Entry_2017.csv'))
        self.data_files = self.df[['Image Index', 'Finding Labels']]
        if test:
            self.image_names = open(join(dir, 'test_list.txt')).readlines()
        else:
            self.image_names = open(join(dir, 'train_val_list.txt')).readlines()
        self.files = dict([(os.path.basename(f), f) for f in glob.iglob(join(dir, '/**/*.png'), recursive=True)])
        print(len(self.files))
    def __len__(self,):
        pass

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image_path = self.files[image_name]
        image_frame = self.data_files.loc[self.data_files['Image Index'] == image_name]
        
        

if __name__ == "__main__":
    dataset = nih_dataset()