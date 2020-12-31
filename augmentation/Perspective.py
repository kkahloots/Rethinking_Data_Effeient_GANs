
import numpy as np
import tensorflow_addons as tfa
import tensorflow as tf

@tf.function
def rotate(images, angles):
    im_shape = tf.shape(images)
    src_height, src_width = tf.unstack(im_shape)[1:3]
    images = tf.pad(images, [[0, 0], [5, 5], [5, 5], [0, 0]], 'REFLECT')
    images = tf.image.resize(images, (src_height, src_width))
    pad_size = tf.cast(
        tf.cast(tf.maximum(src_height, src_width), tf.float32) * (2.0 - 1.0) / 2 + 0.5, tf.int32)  # larger than usual (sqrt(2))
    images = tf.pad(images, [[0, 0], [pad_size] * 2, [pad_size] * 2, [0, 0]], 'REFLECT')

    images = tfa.image.rotate(images, angles*np.pi/180)
    return tf.slice(images, [0, pad_size, pad_size, 0], [-1, src_height, src_width, -1])


@tf.function
def rand_shift(images, ratio=0.125):
    im_shape = tf.shape(images)
    src_height, src_width = tf.unstack(im_shape)[1:3]
    images = tf.pad(images, [[0, 0], [5, 5], [5, 5], [0, 0]], 'REFLECT')
    images = tf.image.resize(images, (src_height, src_width))
    pad_size = tf.cast(
        tf.cast(tf.maximum(src_height, src_width), tf.float32) * (2.0 - 1.0) / 2 + 0.5, tf.int32)  # larger than usual (sqrt(2))
    images = tf.pad(images, [[0, 0], [pad_size] * 2, [pad_size] * 2, [0, 0]], 'REFLECT')

    batch_size = tf.shape(images)[0]
    image_size = tf.shape(images)[1:3]

    shift = tf.cast(tf.cast(image_size, tf.float32) * ratio + 0.5, tf.int32)
    translation_x = tf.random.uniform([batch_size, 1], -shift[0], shift[0] + 1, dtype=tf.int32)
    translation_y = tf.random.uniform([batch_size, 1], -shift[1], shift[1] + 1, dtype=tf.int32)
    grid_x = tf.clip_by_value(tf.expand_dims(tf.range(image_size[0], dtype=tf.int32), 0) + translation_x + 1, 0, image_size[0] + 1)
    grid_y = tf.clip_by_value(tf.expand_dims(tf.range(image_size[1], dtype=tf.int32), 0) + translation_y + 1, 0, image_size[1] + 1)
    images = tf.gather_nd(tf.pad(images, [[0, 0], [1, 1], [0, 0], [0, 0]], 'REFLECT'), tf.expand_dims(grid_x, -1), batch_dims=1)
    images = tf.transpose(tf.gather_nd(tf.pad(tf.transpose(images, [0, 2, 1, 3]), [[0, 0], [1, 1], [0, 0], [0, 0]], 'REFLECT'), tf.expand_dims(grid_y, -1), batch_dims=1), [0, 2, 1, 3])
    return tf.slice(images, [0, pad_size, pad_size, 0], [-1, src_height, src_width, -1])
