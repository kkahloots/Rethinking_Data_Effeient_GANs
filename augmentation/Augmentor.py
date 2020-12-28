import random
import itertools
import tensorflow as tf

import augmentation.Coloring as color_aug
import augmentation.Distortion as distort_aug
import augmentation.Mirror as mirror_aug
import augmentation.Perspective as pres_aug
import augmentation.Photometric as photo_aug
import augmentation.Translation as trans_aug

scales = [a/100 for a in range(80, 121)]

def clone(x):
    return x


def add_random_brightness(x):
    scale = tf.reduce_max(x).numpy() <= 1.0
    if scale:
        x *= 255
    x = photo_aug.random_brightness(x, max_abs_change=100)
    if scale:
        x /= 255
    return x


def add_random_contrast(x):
    scale = tf.reduce_max(x).numpy() <= 1.0
    if scale:
        x *= 255
    x = photo_aug.random_contrast(x)
    if scale:
        x /= 255
    return x


def add_random_saturation(x):
    scale = tf.reduce_max(x).numpy() <= 1.0
    if scale:
        x *= 255
    x = photo_aug.random_saturation(x)
    if scale:
        x /= 255
    return x


def add_additive_shade(x):
    scale = tf.reduce_max(x).numpy() <= 1.0
    if scale:
        x *= 255
    x = photo_aug.additive_shade(x)
    if scale:
        x /= 255
    return x


def transform_color_space(x):
    scale = tf.reduce_max(x).numpy() <= 1.0
    if scale:
        x *= 255
    x = color_aug.color_space_transform(x)
    if scale:
        x /= 255
    return x


def rotate_random(x):
    scale = tf.reduce_max(x).numpy() <= 1.0
    if scale:
        x *= 255
    a = random.randint(-35, 35)
    x = pres_aug.rotate(x, a)
    if scale:
        x /= 255
    return x


def flip_left_right(x):
    scale = tf.reduce_max(x).numpy() <= 1.0
    if scale:
        x *= 255
    x = mirror_aug.flip_left_right(x)
    if scale:
        x /= 255
    return x


def distort_random(x):
    scale = tf.reduce_max(x).numpy() <= 1.0
    if scale:
        x *= 255
    w = int(x.shape[0])
    n = random.randint(5, w // 5 + 1)
    s = random.randint(-5, 5)
    x = distort_aug.distort(x, num_anchors=w // n, perturb_sigma=s)
    if scale:
        x /= 255
    return x


def shift_random(x):
    scale = tf.reduce_max(x).numpy() <= 1.0
    if scale:
        x *= 255

    r = random.choice(scales)
    x = pres_aug.rand_shift(x, ratio=r)
    if scale:
        x /= 255
    return x


def shear_left_right_random(x):
    scale = tf.reduce_max(x).numpy() <= 1.0
    if scale:
        x *= 255

    l = random.choice(scales)
    x = trans_aug.shear_left_right(x, shear_lambda=l)
    if scale:
        x /= 255
    return x


def translate_top_down_random(x):
    scale = tf.reduce_max(x).numpy() <= 1.0
    if scale:
        x *= 255

    l = random.choice(scales)
    x = trans_aug.shear_top_down(x, shear_lambda=l)
    if scale:
        x /= 255
    return x


def shear_right_left_random(x):
    scale = tf.reduce_max(x).numpy() <= 1.0
    if scale:
        x *= 255

    l = random.choice(scales)
    x = trans_aug.shear_left_right(x, shear_lambda=-l)
    if scale:
        x /= 255
    return x

def shear_down_top_random(x):
    scale = tf.reduce_max(x).numpy() <= 1.0
    if scale:
        x *= 255

    l = random.choice(scales)
    x = trans_aug.shear_down_top(x, shear_lambda=-l)
    if scale:
        x /= 255
    return x

def shear_top_down_random(x):
    scale = tf.reduce_max(x).numpy() <= 1.0
    if scale:
        x *= 255

    l = random.choice(scales)
    x = trans_aug.shear_top_down(x, shear_lambda=-l)
    if scale:
        x /= 255
    return x

def skew_left_right_random(x):
    scale = tf.reduce_max(x).numpy() <= 1.0
    if scale:
        x *= 255

    ll = random.choice(scales)
    lr = random.choice(scales)
    x = trans_aug.skew_left_right(x, l_shear_lambda=ll, r_shear_lambda=lr)

    if scale:
        x /= 255
    return x


def skew_top_left_random(x):
    scale = tf.reduce_max(x).numpy() <= 1.0
    if scale:
        x *= 255

    lt = random.choice(scales)
    ll = random.choice(scales)
    x = trans_aug.skew_top_left(x, t_shear_lambda=lt, l_shear_lambda=ll)

    if scale:
        x /= 255
    return x

def skew_down_left(x):
    scale = tf.reduce_max(x).numpy() <= 1.0
    if scale:
        x *= 255

    lt = random.choice(scales) * -1
    ll = random.choice(scales)
    x = trans_aug.skew_top_left(x, t_shear_lambda=lt, l_shear_lambda=ll)

    if scale:
        x /= 255
    return x


def shear_top_right(x):
    scale = tf.reduce_max(x).numpy() <= 1.0
    if scale:
        x *= 255

    lt = random.choice(scales) * -1
    ll = random.choice(scales) * -1
    x = trans_aug.shear_left_down(x, t_shear_lambda=lt, l_shear_lambda=ll)

    if scale:
        x /= 255
    return x


def shear_left_down(x):
    scale = tf.reduce_max(x).numpy() <= 1.0
    if scale:
        x *= 255

    lt = random.choice(scales)
    ll = random.choice(scales)
    x = trans_aug.shear_left_down(x, t_shear_lambda=lt, l_shear_lambda=ll)

    if scale:
        x /= 255
    return x


def skew_left_top_random(x):
    scale = tf.reduce_max(x).numpy() <= 1.0
    if scale:
        x *= 255

    lt = random.choice(scales)
    ll = random.choice(scales)
    x = trans_aug.skew_left_top(x, t_shear_lambda=lt, l_shear_lambda=ll)

    if scale:
        x /= 255
    return x


def skew_top_down_random(x):
    scale = tf.reduce_max(x).numpy() <= 1.0
    if scale:
        x *= 255

    ll = random.choice(scales)
    lr = random.choice(scales)
    x = trans_aug.skew_top_down(x, t_shear_lambda=ll, d_shear_lambda=lr)

    if scale:
        x /= 255
    return x


photo_aug_list = [clone, add_additive_shade, add_random_brightness, add_random_contrast, \
                         add_random_saturation, add_additive_shade, add_random_brightness,
                         clone, add_random_contrast, add_random_saturation]

distort_aug_list = [clone, distort_random, distort_random, distort_random, \
                           distort_random, distort_random, distort_random, \
                           distort_random, distort_random, distort_random]

mirror_aug_list = [clone, flip_left_right, flip_left_right, flip_left_right, \
                          flip_left_right, flip_left_right, flip_left_right, \
                          flip_left_right, flip_left_right, flip_left_right]

pres_aug_list = [clone, shift_random, shift_random, shift_random, \
                        shift_random, shift_random, shift_random, \
                        shift_random, shift_random, shift_random]

color_aug_list = [clone, transform_color_space, transform_color_space, transform_color_space, \
                         transform_color_space, transform_color_space, transform_color_space, \
                         transform_color_space, transform_color_space, transform_color_space]

trans_aug_list = [clone, shear_top_down_random, shear_left_right_random, shear_down_top_random, \
                         shear_right_left_random, skew_left_right_random, skew_top_down_random, \
                         skew_top_left_random, skew_left_top_random, skew_down_left, shear_left_down, shear_top_right ]



augmentation_functions = list(itertools.product(photo_aug_list,
                                                distort_aug_list,
                                                mirror_aug_list,
                                                pres_aug_list,
                                                color_aug_list,
                                                trans_aug_list))

# augmentation_functions = {
#     'mirror': flip_left_right,
#     'random_brightness': add_random_brightness,
#     'random_contrast': add_random_contrast,
#     'random_saturation': add_random_saturation,
#     'additive_shade': add_additive_shade,
#     'color_space_transform': transform_color_space,
#     'random_rotate': rotate_random,
#     'random_distort': distort_random,
#     'random_shift': shift_random,
#     'random_top_down': translate_top_down_random,
#     'random_left_right': translate_left_right_random,
#     'additive_shade_distort': lambda x: add_additive_shade(distort_random(x)),
#     'distort_additive_shade': lambda x: distort_random(add_additive_shade(x)),
#     'photo_distort': lambda x: add_additive_shade(distort_random(x)),
#     'distort_photo': lambda x: distort_random(add_additive_shade(x)),
# }
