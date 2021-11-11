import os
import glob
import math
from typing import Iterable
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras.utils import Sequence
from tensorflow.keras.layers.experimental import preprocessing
import pandas as pd
from skimage.io import imread
import numpy as np

join = os.path.join

resize = lambda size : preprocessing.Resizing(size, size)

global_augmentation = lambda size: K.Sequential([
    resize(size),
    preprocessing.RandomFlip("vertical"),
    preprocessing.RandomZoom(0.9, 0.9)
], name='global_aug') 

local_augementation = lambda size: K.Sequential([
    resize(size),
    preprocessing.RandomFlip("vertical"),
    preprocessing.RandomZoom(0.4, 0.4),
    preprocessing.RandomContrast(0.7),
], name='local_aug')


def label_split(label, label_dict={}):
    one_hot = [0 for _ in range(len(label_dict))]
    if '|' not in label:
        if label not in label_dict:
            label_dict[label] = len(label_dict)
            one_hot.append(1)
        else:
            one_hot[label_dict[label]] = 1
    else:
        for l in label.split('|'):
            if l not in label_dict:
                label_dict[l] = len(label_dict)
                one_hot.append(1)
            else:
                one_hot[label_dict[l]] = 1
    return one_hot

class nih_dataset(Sequence):
    def __init__(self, dir='/home/mramados/data/nih', image_size=224, test=False, bounding_box=False, batch_size=8, global_aug=False, local_aug=False, localiztion_model=None):
        print("Loading CSV")
        self.batch_size = batch_size
        self.bounding_box = bounding_box
        if bounding_box:
            self.df = pd.read_csv(join(dir, 'BBox_List_2017.csv'))
            self.data_files = self.df
        else:
            self.df = pd.read_csv(join(dir, 'Data_Entry_2017.csv'))
            self.data_files = self.df[['Image Index', 'Finding Labels']]            
        
        labels = pd.unique(self.data_files['Finding Labels'])
        self.label_dict = {}
        for label in labels:
            label_split(label, self.label_dict)        
        if test:
            self.image_names = open(join(dir, 'test_list.txt')).readlines()
        else:
            self.image_names = open(join(dir, 'train_val_list.txt')).readlines()
        print("Loading Files Paths")
        self.files = dict()
        for file_path in glob.iglob(join(dir, "**/*.png"), recursive=True):
            if file_path.lower().endswith(".png"):
                self.files[os.path.basename(file_path)] = file_path
        self.global_aug = global_aug
        self.local_aug = local_aug
        self.localizaiton_model = localiztion_model
        self.image_resize = resize(image_size)

    def __len__(self,):
        return math.ceil(len(self.image_names)  / self.batch_size)
    
    def get_logits_size(self):
        return len(self.label_dict)
    


    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image_name = image_name.strip()
        image = imread(self.files[image_name])
        label = []
        if len(image.shape) != 2:
            image = np.add(image, axis=-1) / image.shape[-1]
            
        im_frame = self.data_files.loc[self.data_files['Image Index'] == image_name]
        if not self.bounding_box:
            label_str = im_frame['Finding Labels'].to_list()
            label = label_split(label_str[0], self.label_dict)
        images = np.array(image)
        images = np.expand_dims(images, (0, -1))
        labels = np.array(label)
        images = tf.convert_to_tensor(images)
        images = self.image_resize(images)
        labels = tf.convert_to_tensor(labels)
        return images, labels

    
    