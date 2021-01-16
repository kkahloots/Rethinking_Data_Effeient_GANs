import tensorflow as tf
import numpy as np
import cv2


def _coloring(image, mask):
    radius = image.shape[0]//2
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    image = cv2.inpaint(image, mask, radius, flags=cv2.INPAINT_TELEA)
    return image


def cutout(images, **kwargs):
    #case_true = tf.image.resize(tf.slice(images, [0, 0, 0, 0], [-1, kwargs['height'], 10, -1], name=None),
    #                            (kwargs['height'], kwargs['width']))

    return images * kwargs['mask']
    #condition = tf.equal(images, 0)
    #return tf.where(condition, case_true, images)


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
