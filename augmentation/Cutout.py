import tensorflow as tf
import tensorflow_addons as tfa
from augmentation.Coloring import simplest_cb
import numpy as np
import cv2


def _coloring(image, mask):
    radius = image.shape[0]//2
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    image = cv2.inpaint(image, mask, radius, flags=cv2.INPAINT_TELEA)
    return image


def cutout(images, **kwargs):

    images =  images * kwargs['mask']
    dbatch = dilation2d(images)
    condition = tf.equal(images, 0)
    images = tf.where(condition, dbatch, images)

    def _py_color_space(img):
        images = []
        for i in range(len(img)):
            images += [simplest_cb(cv2.cvtColor(cv2.cvtColor(
                img[i].numpy().astype(np.uint8), cv2.IMREAD_COLOR), cv2.IMREAD_COLOR))]
        return np.array(images)

    return tf.py_function(_py_color_space, [images], tf.float32)


def dilation2d(img4D):
    b, h, w, c = img4D.get_shape().as_list()
    kernel = tf.ones((h//5, h//5, c))
    output4D = tf.nn.dilation2d(img4D, filters=kernel, strides=(1,1,1,1), dilations=(1,1,1,1),
                                data_format='NHWC', padding="SAME")
    output4D = output4D - tf.ones_like(output4D)

    return output4D


def rand_mask(batch_size, height, width, ratio=0.5):
    cutout_size = tf.cast(tf.cast((width, height), tf.float32) * ratio + 0.5, tf.int32)
    offset_x = tf.random.uniform([batch_size, 1, 1], maxval=width + (1 - cutout_size[0] % 2), dtype=tf.int32)
    offset_y = tf.random.uniform([batch_size, 1, 1], maxval=height + (1 - cutout_size[1] % 2), dtype=tf.int32)
    grid_batch, grid_x, grid_y = tf.meshgrid(tf.range(batch_size, dtype=tf.int32), tf.range(cutout_size[0], dtype=tf.int32), tf.range(cutout_size[1], dtype=tf.int32), indexing='ij')
    cutout_grid = tf.stack([grid_batch, grid_x + offset_x - cutout_size[0] // 2, grid_y + offset_y - cutout_size[1] // 2], axis=-1)
    mask_shape = tf.stack([batch_size, width, height])
    cutout_grid = tf.maximum(cutout_grid, 0)
    cutout_grid = tf.minimum(cutout_grid, tf.reshape(mask_shape - 1, [1, 1, 1, 3]))
    mask = tf.maximum(1 - tf.scatter_nd(cutout_grid, tf.ones([batch_size, cutout_size[0], cutout_size[1]], dtype=tf.float32), mask_shape), 0)
    return tf.expand_dims(mask, axis=3)


def patch(images, **kwargs):
    case_true = tf.random.shuffle(images)

    images = images * kwargs['mask']
    condition = tf.equal(images, 0)
    images = tf.where(condition, case_true, images)
    def _py_color_space(img):
        images = []
        for i in range(len(img)):
            images += [simplest_cb(cv2.cvtColor(cv2.cvtColor(
                img[i].numpy().astype(np.uint8), cv2.IMREAD_COLOR), cv2.IMREAD_COLOR))]
        return np.array(images)

    return tf.py_function(_py_color_space, [images], tf.float32)