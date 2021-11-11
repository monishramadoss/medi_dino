from model import VIT_Class
import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential, losses, metrics
from dataset import nih_dataset, resize


class DINO(Model):
    def __init__(self, student, teacher):
        super(DINO, self).__init__()
        self.student = student
        self.teacher = teacher
        self.loss_tracker = metrics.CosineSimilarity()

    @property
    def metrics(self):
        metrics = super().metrics
        metrics.append(self.loss_tracker)
        return metrics
    
    def compile(self, optimizer, metrics, dislation_loss_fn, temperature):
        super(DINO, self).compile(optimizer=optimizer, metrics=metrics)
        self.dislation_loss_fn = dislation_loss_fn
        self.temperature = temperature
    
    def train_step(self, data):
        x, _ = data

        teacher_logit = self.teacher(x, training=False)
        with tf.GradientTape() as tape:
            student_logit = self.student(x, training=True)
            distillation_loss = self.dislation_loss_fn(teacher_logit, student_logit)

        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(distillation_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.loss_tracker.update_state(distillation_loss)
        return {"distillation_loss": self.loss_tracker.result()}
    
    def test_step(self, data):
        x, _ = data

        