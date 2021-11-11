import tensorflow as tf
from train import DINO
from model import VIT_Class
from dataset import nih_dataset


dataset = nih_dataset()
valid_dataset = nih_dataset(test=False)

VIT_Class(224, 16, logits_size=dataset.get_logits_size())

model = VIT_Class()
model.compile(optimizer='adam', loss='sce', metrics='accuracy')
model.fit(dataset, validation_data=valid_dataset, epochs=100, validation_freq=10, )

