import os
import glob
import math
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras.utils import Sequence
import pandas as pd
from skimage.io import imread


join = os.path.join


class nih_dataset(Sequence):
    def __init__(self, dir='/home/mramados/data/nih', test=False, bounding_box=False, batch_size=8):
        self.batch_size = batch_size
        self.bounding_box = bounding_box
        if bounding_box:
            self.df = pd.read_csv(join(dir, 'BBox_List_2017.csv'))
            self.data_files = self.df[['Image Index', 'Finding Labels']]            
        else:
            self.df = pd.read_csv(join(dir, 'Data_Entry_2017.csv'))
            self.data_files = self.df
        
        self.classification_label = dict([(v, i) for v, i in enumerate(pd.unique(self.data_files['Finding Labels']))])
        
        if test:
            self.image_names = open(join(dir, 'test_list.txt')).readlines()
        else:
            self.image_names = open(join(dir, 'train_val_list.txt')).readlines()
       
        self.files = dict([(os.path.basename(f), f) for f in glob.iglob(join(dir, '/**/*.png'), recursive=True)])
        print(len(self.files))
        
    def __len__(self,):
        return math.ceil(len(self.image_names)  / self.batch_size)

    def __getitem__(self, idx):
        image_names = self.image_names[idx]
        images = []
        labels = []
        
        for image_path in image_names:
            image = imread(self.files[image_name])
            im_frame = self.data_files.loc[self.data_files['Image Index'] == image_name]
            if not self.bounding_box:
                label = classification_label[im_fram['Finding Labels']]
                labels.append(label)
            images.append(image)
        
        return np.array(image), np.array(labels)

if __name__ == "__main__":
    dataset = nih_dataset()
    print(dataset[0][0].shape)