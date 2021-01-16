import tensorflow as tf

def flip_left_right(images, **kwargs):
    return tf.image.flip_left_right(images)
