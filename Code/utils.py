"""
This file contains utility functions.
"""

import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import numpy as np


# Enable dynamic memory allocation
def enable_gpu_mem_growth():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


class Trainer(object):
    # Define training step
    @tf.function
    def train_step(self, x, y, mdl, opt):
        with tf.GradientTape() as tape:
            logits = mdl(x, training=True)
            loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y, logits, from_logits=True))
        grads = tape.gradient(loss, mdl.trainable_variables)
        opt.apply_gradients(zip(grads, mdl.trainable_variables))
        return loss

    # Define training step for bias correction
    def bias_correction_step(self, x, y, mdl, opt):
        with tf.GradientTape() as tape:
            logits = mdl(x, training=True)
            loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y, logits, from_logits=True))
        grads = tape.gradient(loss, (mdl.w, mdl.b))
        opt.apply_gradients(zip(grads, (mdl.w, mdl.b)))
        return loss

    # Define training and distillation step
    @tf.function
    def train_distill_step(self, x, y, x_r, y_r, l_r, mdl, opt, task, T=2.0):
        with tf.GradientTape() as tape:
            # Compute classification loss
            logits = mdl(x, training=True)
            logits_r = mdl(x_r, training=True)
            class_loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(tf.concat((y, y_r), axis=0), tf.concat((logits, logits_r), axis=0), from_logits=True))
            # Compute distillation loss
            mask = tf.math.less(tf.range(0, y.shape[-1]), tf.multiply(tf.constant(2, dtype=tf.int32), task))
            masked_l_r = tf.boolean_mask(tf.divide(l_r, tf.constant(T, dtype=tf.float32)), mask, axis=1)
            masked_logits_r = tf.boolean_mask(tf.divide(logits_r, tf.constant(T, dtype=tf.float32)), mask, axis=1)
            dist_loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(tf.nn.softmax(masked_l_r), masked_logits_r, from_logits=True))
            # Combine losses
            num_old_classes = tf.multiply(tf.constant(2.0, dtype=tf.float32), task)
            weight = tf.divide(num_old_classes, tf.add(num_old_classes, tf.constant(2.0, dtype=tf.float32)))
            loss = tf.add(tf.multiply(tf.subtract(tf.constant(1.0, dtype=tf.float32), weight), class_loss), tf.multiply(weight, dist_loss))
        grads = tape.gradient(loss, mdl.trainable_variables)
        opt.apply_gradients(zip(grads, mdl.trainable_variables))
        return loss


# Standardization
def standardize(batch):
    x = tf.divide(tf.cast(batch["image"], tf.float32), tf.constant(255.0, tf.float32))
    y = tf.one_hot(tf.cast(batch["label"], tf.int32), 10)
    return x, y


def combine_img(x_plt):
    dim = int(np.ceil(np.sqrt(x_plt.shape[0])))
    img = np.zeros((x_plt.shape[1] * dim, x_plt.shape[2] * dim, x_plt.shape[3]))
    for i in range(dim):
        for j in range(dim):
            idx = j * dim + i
            if idx < x_plt.shape[0]:
                img[i * x_plt.shape[1]:(i + 1) * x_plt.shape[1], j * x_plt.shape[2]:(j + 1) * x_plt.shape[1]] = x_plt[idx].numpy()
            else:
                img[i * x_plt.shape[1]:(i + 1) * x_plt.shape[1], j * x_plt.shape[2]:(j + 1) * x_plt.shape[1]] = np.zeros_like(x_plt[0])
    return img


def plot(x_plt, name, path):
    img = combine_img(x_plt)
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.savefig(fname=path+"/{}.png".format(name), format="png")


def reinitialize_model(mdl):
    for layer in mdl.layers:
        if layer.trainable:
            if isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.Dense):
                if hasattr(layer, "kernel"):
                    layer.kernel.assign(layer.kernel_initializer(layer.kernel.shape))
                if hasattr(layer, "bias"):
                    if layer.bias is not None:
                        layer.bias.assign(layer.bias_initializer(layer.bias.shape))
            if isinstance(layer, tfa.layers.InstanceNormalization) or isinstance(layer, tf.keras.layers.BatchNormalization):
                if hasattr(layer, "gamma"):
                    layer.gamma.assign(layer.gamma_initializer(layer.gamma.shape))
                if hasattr(layer, "beta"):
                    layer.beta.assign(layer.beta_initializer(layer.beta.shape))
