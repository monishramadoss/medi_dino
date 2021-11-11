import tensorflow as tf
from tensorflow.keras import Model, losses, metrics
import numpy as np

class DINOLoss(losses.Loss):
    def __init__(self, nepochs, student_temp, teacher_temp, warmup_teacher_temp_epochs, warmup_teacher_temp, center_momentum, ncrops, out_dims):
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.center =  tf.zeros([1, out_dims])
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        self.teacher_temp_schedule = np.concatenate([
            np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ])
        self.loss = self.loss_obj
        self.epoch = 0

    def call(self, student_logit, teacher_logit, sample_weight=None):
        self.teacher_temp = self.teacher_temp_schedule[self.epoch]
        student_out = student_logit / self.student_temp
        teacher_out = tf.nn.softmax((teacher_logit - self.center) / self.teacher_temp, axis=-1)
        total_loss = 0.0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(student_out.shape[0]):
                if v == q:
                    continue
                loss = tf.math.reduce_sum(-q * tf.nn.log_softmax(student_out[v], axis=-1), axis=-1)
                n_loss_terms += 1
                total_loss += loss.mean()

        total_loss /= n_loss_terms
        self.update_center(teacher_out)
        self.epoch += 1
    def update_center(self, teacher_logit):
        batch_center = tf.math.reduce_sum(teacher_logit, axis=0, keepdim=True)
        batch_center = batch_center / (len(teacher_logit))
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class DINO(Model):
    def __init__(self, student, teacher, nepochs, teacher_temp,  warmup_teacher_temp, warmup_teacher_temp_epochs):
        super(DINO, self).__init__()
        self.student = student
        self.teacher = teacher
        self.loss_tracker = metrics.CosineSimilarity()
        self.teacher_temp_schedule = np.concatenate([
            np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ])

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
        x, y = data
        student_logit = self.student(x, training=False)
        self.compiled_metrics.update_state(y, student_logit)
        return {m.name: m.result() for m in self.metrics}
