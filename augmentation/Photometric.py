#
# Photometric augmentations

import tensorflow as tf
import tensorflow_addons as tfa
from augmentation.Coloring import simplest_cb
import cv2
import numpy as np

def random_brightness(images, **kwargs):
    images = images + kwargs['magnitude']
    return adjust_color(tf.clip_by_value(images, 0, 255))


def random_saturation(images,  **kwargs):
    images_mean = tf.reduce_mean(images, axis=3, keepdims=True)
    images = (images - images_mean) * kwargs['magnitude'] + images_mean
    return adjust_color(tf.clip_by_value(images, 0, 255))


def random_contrast(images, **kwargs):
    images_mean = tf.reduce_mean(images, axis=[1, 2, 3], keepdims=True)
    images = (images - images_mean) * kwargs['magnitude'] + images_mean
    return adjust_color(tf.clip_by_value(images, 0, 255))

def adjust_color(images):
    def _py_color(img):
        images = []
        for i in range(len(img)):
            images += [simplest_cb(cv2.cvtColor(
                img[i].numpy().astype(np.uint8), cv2.IMREAD_COLOR))]
        return np.array(images)

    return tf.py_function(_py_color, [images], tf.float32)
