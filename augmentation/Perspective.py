
import numpy as np
import tensorflow_addons as tfa
import tensorflow as tf

def rotate(images, **kwargs):
    images = tf.pad(images, [[0, 0], [5, 5], [5, 5], [0, 0]], 'REFLECT')
    images = tf.image.resize(images, (kwargs['height'], kwargs['width']))
    pad_size = tf.cast(
        tf.cast(tf.maximum(kwargs['height'], kwargs['width']), tf.float32) * (2.0 - 1.0) / 2 + 0.5, tf.int32)  # larger than usual (sqrt(2))
    images = tf.pad(images, [[0, 0], [pad_size] * 2, [pad_size] * 2, [0, 0]], 'REFLECT')

    images = tfa.image.rotate(images, kwargs['angles']*np.pi/180)
    return tf.slice(images, [0, pad_size, pad_size, 0], [-1, kwargs['height'], kwargs['width'], -1])


def rand_shift(images, **kwargs):

    images = tf.pad(images, [[0, 0], [5, 5], [5, 5], [0, 0]], 'REFLECT')
    images = tf.image.resize(images, (kwargs['height'], kwargs['width']))
    pad_size = tf.cast(
        tf.cast(tf.maximum(kwargs['height'], kwargs['width']), tf.float32) * (2.0 - 1.0) / 2 + 0.5, tf.int32)  # larger than usual (sqrt(2))
    images = tf.pad(images, [[0, 0], [pad_size]*2, [pad_size]*2, [0, 0]], 'REFLECT')

    translation_x = kwargs['translation_x']
    translation_y = kwargs['translation_y']
    grid_x = tf.clip_by_value(tf.expand_dims(tf.range(kwargs['pwidth'], dtype=tf.int32), 0) + translation_x + 1, 0, kwargs['pwidth'] + 1)
    grid_y = tf.clip_by_value(tf.expand_dims(tf.range(kwargs['pheight'], dtype=tf.int32), 0) + translation_y + 1, 0, kwargs['pheight'] + 1)
    images = tf.gather_nd(tf.pad(images, [[0, 0], [1, 1], [0, 0], [0, 0]], 'REFLECT'), tf.expand_dims(grid_x, -1), batch_dims=1)
    images = tf.transpose(tf.gather_nd(tf.pad(tf.transpose(images, [0, 2, 1, 3]), [[0, 0], [1, 1], [0, 0], [0, 0]], 'REFLECT'), tf.expand_dims(grid_y, -1), batch_dims=1), [0, 2, 1, 3])
    return tf.slice(images, [0, pad_size, pad_size, 0], [-1, kwargs['height'], kwargs['width'], -1])
