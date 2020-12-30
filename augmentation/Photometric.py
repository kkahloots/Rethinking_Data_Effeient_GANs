#
# Photometric augmentations
# Some codes come from  https://github.com/rpautrat/SuperPoint
# input image is supposed to be 3D tensor [H,W,C] and floating 0~255 values

import cv2
import numpy as np
import tensorflow as tf


def random_brightness(image, max_abs_change=50):
    return tf.clip_by_value(tf.image.random_brightness(image, max_abs_change), 0, 255)


def random_saturation(image, strength_range=[0.5, 1.5]):
    return tf.clip_by_value(tf.image.random_saturation(image, *strength_range), 0, 255)


def random_contrast(image, strength_range=[0.5, 1.5]):
    return tf.clip_by_value(tf.image.random_contrast(image, *strength_range), 0, 255)


def additive_shade(images, nb_ellipses=20, transparency_range=[-0.5, 0.8],
                   kernel_size_range=[250, 350]):
    def _py_additive_shade(images):
        shaded_images = []
        for img in images:
            min_dim = min(img.shape[:2]) / 4
            mask = np.zeros(img.shape[:2], np.uint8)
            for i in range(nb_ellipses):
                ax = int(max(np.random.rand() * min_dim, min_dim / 5))
                ay = int(max(np.random.rand() * min_dim, min_dim / 5))
                max_rad = max(ax, ay)
                x = np.random.randint(max_rad, img.shape[1] - max_rad)  # center
                y = np.random.randint(max_rad, img.shape[0] - max_rad)
                angle = np.random.rand() * 90
                cv2.ellipse(mask, (x, y), (ax, ay), angle, 0, 360, 255, -1)

            transparency = np.random.uniform(*transparency_range)
            kernel_size = np.random.randint(*kernel_size_range)
            if (kernel_size % 2) == 0:  # kernel_size has to be odd
                kernel_size += 1
            mask = cv2.GaussianBlur(mask.astype(np.float32), (kernel_size, kernel_size), 0)
            shaded = img * (1 - transparency * mask[..., np.newaxis] / 255.)
            shaded_images += [np.clip(shaded, 0, 255)]
        return np.stack(shaded_images)

    shaded = tf.py_function(_py_additive_shade, [images], tf.float32)
    res = tf.reshape(shaded, tf.shape(images))
    return res
