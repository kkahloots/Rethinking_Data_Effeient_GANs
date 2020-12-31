import tensorflow as tf

def flip_left_right(images, batch_shape=None):
    return tf.image.random_flip_left_right(images)
