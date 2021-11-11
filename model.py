import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential, losses
import tensorflow.keras as K
import tensorflow as tf
import numpy as np

def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)
    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule

def mlp( hidden_units, dropout_rate):
    mlp_lst = []    
    for units in hidden_units:
        mlp_lst += [
            layers.Dense(units, activation=tf.nn.gelu),
            layers.Dropout(dropout_rate)
        ]    
    return Sequential(mlp_lst)

class PatchExtract(layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super(PatchExtract, self).__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=(1, self.patch_size, self.patch_size, 1),
            strides=(1, self.patch_size, self.patch_size, 1),
            rates=(1, 1, 1, 1),
            padding="VALID",
        )
        patch_dim = patches.shape[-1]
        patch_num = patches.shape[1]
        _shape = tf.convert_to_tensor([-1, patch_num ** 2, patches.shape[-1]],)
        return tf.reshape(patches, shape=_shape)


class PatchEmbedding(layers.Layer):
    def __init__(self, num_patch, embed_dim, **kwargs):
        super(PatchEmbedding, self).__init__(**kwargs)
        self.num_patch = num_patch
        self.proj = layers.Dense(embed_dim)
        self.pos_embed = layers.Embedding(input_dim=num_patch, output_dim=embed_dim)

    def call(self, patch):
        pos = tf.range(start=0, limit=self.num_patch, delta=1)
        return self.proj(patch) + self.pos_embed(pos)

class EncoderBlock(layers.Layer):
    def __init__(self, num_heads, projection_dim, atten_dropout=0.1, mlp_dropout=0.1):
        super(EncoderBlock, self).__init__()
        self.l1 = layers.LayerNormalization(epsilon=1e-6)
        self.l2 = layers.LayerNormalization(epsilon=1e-6)
        self.atten = layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim, dropout=atten_dropout)
        self.a1 = layers.Add()
        self.mlp = mlp([projection_dim * 2, projection_dim], mlp_dropout)
        self.a2 = layers.Add()
        self.mlp_dropout = mlp_dropout

    def call(self, x):
        x1 = self.l1(x)
        x2 = self.atten(x1, x1)
        x3 = self.a1([x2, x])
        x4 = self.l2(x3)
        x5 = self.mlp(x4)
        x6 = self.a2([x5, x3])
        return x6

class VIT(Model):
    def __init__(self, image_size, patch_size, projection_dim=64, num_heads=4, transformer_layers=8, atten_dropout=0.1, mlp_dropout=0.1):
        super(VIT, self).__init__()
        self.patches = PatchExtract(patch_size, )
        num_patches = (image_size // patch_size) ** 2
        print("num_patches: ", num_patches)
        self.patch_encoder = PatchEmbedding(num_patches, projection_dim)
        self.transformer_blocks = [EncoderBlock(num_heads, projection_dim, atten_dropout, mlp_dropout) for _ in range(transformer_layers)]
        self.l1 = layers.LayerNormalization(epsilon=1e-6)
    
    def call(self, x):
        patches = self.patches(x)
        embedd = self.patch_encoder(patches)
        for block in self.transformer_blocks:
            embedd = block(embedd)
        return embedd

class VIT_Class(Model):
    def __init__(self, image_size, patch_size, logits_size):
        super(VIT_Class, self).__init__()
        self.vit = VIT(image_size, patch_size)
        self.dense_0 = layers.Dense(logits_size ** 2)
        self.dense_1 = layers.Dense(logits_size)
        
    def call(self, x):
        y = self.vit(x)
        y = self.dense_0(y)
        return self.dense_1(y)
    



    
    
    