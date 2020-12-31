import random
import cv2
import numpy as np
import tensorflow as tf

flags = sorted(list(range(0, 8)) + list(range(32, 36)) + list(range(60, 62)) + list(range(72, 74)) + \
               list(range(10, 12)))

def color_space_transform(images, batch_shape=None):
    def _py_color_space(img):
        images = []
        for i in range(len(img)):
            f = random.choice(flags)
            images += [cv2.cvtColor(cv2.cvtColor(cv2.cvtColor(
                img[i].numpy().astype(np.uint8), cv2.IMREAD_COLOR), f), cv2.IMREAD_COLOR)]
        return np.array(images)

    return tf.py_function(_py_color_space, [images], tf.float32)