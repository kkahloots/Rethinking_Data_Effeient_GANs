import tensorflow as tf
import tensorflow_addons as tfa
from augmentation.Coloring import adjust_color
import numpy as np
import cv2



def cutout(images, **kwargs):

    images =  images * kwargs['mask']
    #dbatch = dilation2d(images)
    #condition = tf.equal(images, 0)
    #images = tf.where(condition, dbatch, images)
    #images = tfa.image.equalize(images)
    return inpaint(images, kwargs['mask'])

def patch(images, **kwargs):
    case_true = tf.random.shuffle(images)

    images = images * kwargs['mask']
    condition = tf.equal(images, 0)
    images = tf.where(condition, case_true, images)
    images = tfa.image.equalize(images)
    return adjust_color(tfa.image.equalize(images),5)



def inpaint(images, masks):
    def _py_inpaint(images, masks):
        imgs = []
        radius = images.shape[1]//10
        for i in range(len(images)):
            msk = masks[i].numpy()
            ix = np.where(msk == 0)
            msk[ix] = 255
            ix = np.where(msk == 1)
            msk[ix] = 0

            msk  = cv2.cvtColor(cv2.cvtColor(msk.astype(np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY)
            imgs += [cv2.inpaint(cv2.cvtColor(
                images[i].numpy().astype(np.uint8), cv2.IMREAD_COLOR), msk, radius, flags=cv2.INPAINT_TELEA)]
        return np.array(imgs)

    return tf.py_function(_py_inpaint, [images, masks], tf.float32)


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
