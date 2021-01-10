import tensorflow as tf

def flip_left_right(images, batch_shape=None):
    return tf.image.flip_left_right(images)
