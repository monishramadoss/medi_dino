from model import VIT_Class
from tensorflow.keras import layers, Model, Sequential, losses
from dataset import nih_dataset, resize



def student_warmup():    
    dataset = nih_dataset()
    image_size = 256
    model = VIT_Class(image_size, 32, dataset.get_logits_size(), resize(image_size))
    model.compile(optimizer='Adam', loss='mse', metrics=['accuracy'])
    model.fit(dataset, batch_size=8, epochs=1, verbose=1)
    


if __name__ == '__main__':
    student_warmup()