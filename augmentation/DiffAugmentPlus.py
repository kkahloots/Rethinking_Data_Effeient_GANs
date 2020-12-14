import random

import cv2
import numpy as np
import tensorflow as tf
from Augmentor.Operations import Skew, Distort, Shear, Flip, Zoom, Rotate, RotateStandard
from PIL import Image

from augmentation.Cloner import Clone
from augmentation.Colorizer import Colorize
from augmentation.Skitcher import Skitch
from augmentation.Patcher import Patcher
from augmentation.Rotate_Zoom import RotateZoom

cycle_epoch = int(1e4)
max_level = 5

def Augment(x, epoch=0):
    level = min(epoch//cycle_epoch, max_level)
    return  DiffAugmentPlus(x, level=level)


def DiffAugmentPlus(x, level=1):
    aug_x = None
    while aug_x is None:
        try:
            fns = []
            for _ in range(level):
                fns += [AUGMENT_FNS[random.choice([*AUGMENT_FNS.keys()])]]
            for f in fns:
                #print(f)
                x = f(x)
            aug_x = 1
        except:
            pass

    return x


def rand_brightness(x):
    magnitude = tf.random.uniform([tf.shape(x)[0]//2, 1, 1, 1]) - 0.5
    x = x + magnitude
    return x


def rand_saturation(x):
    magnitude = tf.random.uniform([tf.shape(x)[0]//2, 1, 1, 1]) * 2
    x_mean = tf.reduce_mean(x, axis=3, keepdims=True)
    x = (x - x_mean) * magnitude + x_mean
    return x


def rand_contrast(x):
    magnitude = tf.random.uniform([tf.shape(x)[0]//2, 1, 1, 1]) + 0.5
    x_mean = tf.reduce_mean(x, axis=[1, 2, 3], keepdims=True)
    x = (x - x_mean) * magnitude + x_mean
    return x


def rand_translation(x, ratio=0.125):
    batch_size = tf.shape(x)[0]
    image_size = tf.shape(x)[1:3]
    shift = tf.cast(tf.cast(image_size, tf.float32) * ratio + 0.5, tf.int32)
    translation_x = tf.random.uniform([batch_size, 1], -shift[0], shift[0] + 1, dtype=tf.int32)
    translation_y = tf.random.uniform([batch_size, 1], -shift[1], shift[1] + 1, dtype=tf.int32)
    grid_x = tf.clip_by_value(tf.expand_dims(tf.range(image_size[0], dtype=tf.int32), 0) + translation_x + 1, 0,
                              image_size[0] + 1)
    grid_y = tf.clip_by_value(tf.expand_dims(tf.range(image_size[1], dtype=tf.int32), 0) + translation_y + 1, 0,
                              image_size[1] + 1)
    x = tf.gather_nd(tf.pad(x, [[0, 0], [1, 1], [0, 0], [0, 0]]), tf.expand_dims(grid_x, -1), batch_dims=1)
    x = tf.transpose(tf.gather_nd(tf.pad(tf.transpose(x, [0, 2, 1, 3]), [[0, 0], [1, 1], [0, 0], [0, 0]]),
                                  tf.expand_dims(grid_y, -1), batch_dims=1), [0, 2, 1, 3])
    return x


def rand_cutout(x, ratio=0.25):
    batch_size = tf.shape(x)[0]
    image_size = tf.shape(x)[1:3]
    cutout_size = tf.cast(tf.cast(image_size, tf.float32) * ratio + 0.5, tf.int32)
    offset_x = tf.random.uniform([tf.shape(x)[0], 1, 1], maxval=image_size[0] + (1 - cutout_size[0] % 2),
                                 dtype=tf.int32)
    offset_y = tf.random.uniform([tf.shape(x)[0], 1, 1], maxval=image_size[1] + (1 - cutout_size[1] % 2),
                                 dtype=tf.int32)
    grid_batch, grid_x, grid_y = tf.meshgrid(tf.range(batch_size, dtype=tf.int32),
                                             tf.range(cutout_size[0], dtype=tf.int32),
                                             tf.range(cutout_size[1], dtype=tf.int32), indexing='ij')
    cutout_grid = tf.stack(
        [grid_batch, grid_x + offset_x - cutout_size[0] // 2, grid_y + offset_y - cutout_size[1] // 2], axis=-1)
    mask_shape = tf.stack([batch_size, image_size[0], image_size[1]])
    cutout_grid = tf.maximum(cutout_grid, 0)
    cutout_grid = tf.minimum(cutout_grid, tf.reshape(mask_shape - 1, [1, 1, 1, 3]))
    mask = tf.maximum(
        1 - tf.scatter_nd(cutout_grid, tf.ones([batch_size, cutout_size[0], cutout_size[1]], dtype=tf.float32),
                          mask_shape), 0)
    x = x * tf.expand_dims(mask, axis=3)
    return x


def tranform(func, images):
    cv_images = [cv2.cvtColor((image * 255).astype(np.uint8), cv2.IMREAD_COLOR) for image in images]
    dim = cv_images[0].shape[:2]

    if str(func) in ['Skew', 'Zoom', 'Rotate']:
        top, bottom = random.randint(0, images.shape[0] // 2 + 1), random.randint(0, images.shape[0] // 2 + 1)
        left, right = random.randint(0, images.shape[0] // 2 + 1), random.randint(0, images.shape[0] // 2 + 1)

        for i in range(len(cv_images)):
            padding = random.choice([cv2.BORDER_REPLICATE, cv2.BORDER_REFLECT, cv2.BORDER_REFLECT_101, cv2.BORDER_CONSTANT])
            cv_images[i] = cv2.copyMakeBorder(cv_images[i], top, bottom, left, right, padding)

    images = func.perform_operation([Image.fromarray(cv_image) for cv_image in cv_images])

    for i in range(len(images)):
        if (images[i].width, images[i].height) != dim:
            images[i] = images[i].resize(dim)

    return np.array([np.array(image) for image in images])/ 255.0


def rand_skew(x):
    return tf.cast(tranform(Skew(probability=1, skew_type="RANDOM", magnitude=0.7),
                            x.numpy() ), tf.float32)


def rand_colorize(x):
    return tf.cast(tranform(Colorize(probability=1), x.numpy()), tf.float32)


def rand_distort(x):
    return tf.cast(tranform(Distort(probability=1, grid_width=random.randint(1, x.shape[0]//5 + 1),
                                            grid_height=random.randint(1, x.shape[0]//5 + 1),
                                            magnitude=random.randint(1, 10)), x.numpy()), tf.float32)


def rand_shear(x):
    return tf.cast(tranform(Shear(probability=1, max_shear_left=0, max_shear_right=random.randint(5, 15)) \
                  if random.randint(0, 1) == 1 else Shear(probability=1, max_shear_left=random.randint(5, 15),
                                                          max_shear_right=0), x.numpy()), tf.float32)


def clone(x):
    return tf.cast(tranform(Clone(probability=1), x.numpy()), tf.float32)


def rand_flip(x):
    return tf.cast(tranform(Flip(probability=1, top_bottom_left_right="RANDOM"), x.numpy()),
                   tf.float32)

def left_right_flip(x):
    return tf.cast(tranform(Flip(probability=1, top_bottom_left_right="LEFT_RIGHT"), x.numpy()),
                   tf.float32)

def skitch(x):
    return tf.cast(tranform(Skitch(probability=1), x.numpy()), tf.float32)


def rand_rotate(x):
    return tf.cast(tranform(Rotate(probability=1, rotation=random.randint(1, 360)),
                            x.numpy()), tf.float32)


def rand_rotateZoom(x):
    return tf.cast(tranform(RotateZoom(probability=1, rotation=random.randint(1, 360)),
                            x.numpy()), tf.float32)


def rand_rotate(x):
    return tf.cast(tranform(Rotate(probability=1, rotation=random.randint(1, 360)),
                            x.numpy()), tf.float32)

def rand_rotate_stand(x):
    return tf.cast(tranform(RotateStandard(probability=1, max_left_rotation=random.randint(1, 360), max_right_rotation=random.randint(1, 360), expand=False),
                            x.numpy()), tf.float32)

def rand_zoom(x):
    return tf.cast(tranform(Zoom(probability=1, min_factor=random.randint(5, 10) // 10,
                                         max_factor=random.randint(10, 15) // 10), x.numpy()), tf.float32)


def rand_patch(x):
    return tf.cast(tranform(Patcher(probability=1), x.numpy()), tf.float32)


AUGMENT_FNS = {
    'clone1': clone,
    'clone2': clone,
    'clone3': clone,
    'clone4': clone,
    'clone5': clone,
    'patch': rand_patch,
    'distort': rand_distort,
    'flip': rand_flip,
    'flip1': rand_flip,
    'lr_flip': left_right_flip,
    'lr_flip1': left_right_flip,
    'lr_flip2': left_right_flip,
    'brightness': rand_brightness,
    'saturation': rand_saturation,
    'contrast': rand_contrast,
    #'skitch': skitch,
    'colorize1': rand_colorize,
    'colorize2': rand_colorize,
    'colorize3': rand_colorize,
    'skew': rand_skew,
    'shear': rand_shear,
    'rotate': rand_rotate,
    'rotateStand': rand_rotate_stand,
    'rotateZoom': rand_rotateZoom,
    'zoom': rand_zoom,
    #'cutout': rand_cutout,
}
