import tensorflow as tf

def flip_left_right(image):
    return tf.image.random_flip_left_right(image)
