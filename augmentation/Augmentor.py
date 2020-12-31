import random
import itertools
import tensorflow as tf

import augmentation.Coloring as color_aug
import augmentation.Distortion as distort_aug
import augmentation.Mirror as mirror_aug
import augmentation.Perspective as pres_aug
import augmentation.Photometric as photo_aug
import augmentation.Translation as trans_aug
import augmentation.ThreeDimension_Presective as td_pres_aug
import numpy as np

scales = [a/1000 for a in range(80, 121)]
d_scales = [a/100 for a in range(150, 301)]
td_scales = [a/100 for a in range(80, 121)]


def AugmentObject(x, td_prob=0.3):
    aug_3d = np.random.choice([False, True], p=[1-td_prob, td_prob])
    if aug_3d:
        done = True
        functions_list = random.choice(augmentation3d_functions)
        fn = random.choice(functions_list)
        while done:
            fn = random.choice(functions_list)
            if fn in photo_aug_list:
                continue
            else:
                done = False
        aug_patch_fn = lambda x: td_pres_aug.aug_bg_patches(x, td_scales, fn)
        functions_list = [aug_patch_fn if f==fn else f for f in functions_list]
    else:
        functions_list = random.choice(augmentation_functions)

    for f in functions_list:
        x = f(x)

    return x


def AugmentNature(x, td_prob=0.3):
    aug_3d = np.random.choice([False, True], p=[1-td_prob, td_prob])
    if aug_3d:
        done = True
        functions_list = random.choice(augmentation3d_functions)
        fn = random.choice(functions_list)
        while done:
            fn = random.choice(functions_list)
            if fn in photo_aug_list:
                continue
            else:
                done = False
        aug_patch_fn = lambda x: td_pres_aug.aug_bg_patches(x, td_scales, fn)
        functions_list = [aug_patch_fn if f==fn else f for f in functions_list]
    else:
        functions_list = random.choice(nature_augmentation_functions)

    for f in functions_list:
        x = f(x)

    return x


def clone(x):
    return x


def add_random_brightness(x):
    x = photo_aug.random_brightness(x, max_abs_change=100)
    return x


def add_random_contrast(x):
    x = photo_aug.random_contrast(x)
    return x


def add_random_saturation(x):
    x = photo_aug.random_saturation(x)
    return x


def add_additive_shade(x):
    x = photo_aug.additive_shade(x)
    return x


def transform_color_space(x):
    x = color_aug.color_space_transform(x)
    return x


def rotate_random(x):
    a = random.randint(-35, 35)
    x = pres_aug.rotate(x, a)
    return x


def flip_left_right(x):
    x = mirror_aug.flip_left_right(x)
    return x


def distort_random(x):
    n = random.randint(5, 15)
    s = random.randint(-5, 5)
    x = distort_aug.distort(x, num_anchors=n, perturb_sigma=s)

    return x


def shift_random(x):
    r = random.choice(scales)
    x = pres_aug.rand_shift(x, ratio=r)
    return x


def shear_left_right_random(x):
    l = random.choice(scales)
    x = trans_aug.shear_left_right(x, shear_lambda=l)
    return x


def translate_top_down_random(x):
    l = random.choice(scales)
    x = trans_aug.shear_top_down(x, shear_lambda=l)
    return x


def shear_right_left_random(x):
    l = random.choice(scales)
    x = trans_aug.shear_left_right(x, shear_lambda=-l)
    return x

def shear_down_top_random(x):

    l = random.choice(scales)
    x = trans_aug.shear_down_top(x, shear_lambda=-l)
    return x

def shear_top_down_random(x):
    l = random.choice(scales)
    x = trans_aug.shear_top_down(x, shear_lambda=-l)
    return x

def skew_left_right_random(x):

    ll = random.choice(scales)
    lr = random.choice(scales)
    x = trans_aug.skew_left_right(x, l_shear_lambda=ll/random.choice(d_scales), r_shear_lambda=lr/random.choice(d_scales))

    return x


def skew_top_left_random(x):

    lt = random.choice(scales)
    ll = random.choice(scales)
    x = trans_aug.skew_top_left(x, t_shear_lambda=lt/random.choice(d_scales), l_shear_lambda=ll/random.choice(d_scales))

    return x

def skew_down_left(x):

    lt = random.choice(scales) * -1
    ll = random.choice(scales)
    x = trans_aug.skew_top_left(x, t_shear_lambda=lt/random.choice(d_scales), l_shear_lambda=ll/random.choice(d_scales))

    return x


def shear_top_right(x):


    lt = random.choice(scales) * -1
    ll = random.choice(scales) * -1
    x = trans_aug.shear_left_down(x, t_shear_lambda=lt/random.choice(d_scales), l_shear_lambda=ll/random.choice(d_scales))


    return x


def shear_left_down(x):

    lt = random.choice(scales)
    ll = random.choice(scales)
    x = trans_aug.shear_left_down(x, t_shear_lambda=lt/random.choice(d_scales), l_shear_lambda=ll/random.choice(d_scales))

    return x


def skew_left_top_random(x):

    lt = random.choice(scales)
    ll = random.choice(scales)
    x = trans_aug.skew_left_top(x, t_shear_lambda=lt/random.choice(d_scales), l_shear_lambda=ll/random.choice(d_scales))


    return x


def skew_top_down_random(x):

    ll = random.choice(scales)
    lr = random.choice(scales)
    x = trans_aug.skew_top_down(x, t_shear_lambda=ll/random.choice(d_scales), d_shear_lambda=lr/random.choice(d_scales))

    return x


# def resize_patches_random(x):
#     # scale = tf.reduce_max(x).numpy() <= 1.0
#     # if scale:
#     #     x *= 255
#
#     scales = [a / 100 for a in range(80, 120)]
#     x = td_pres_aug.resize_patches(x, scales)
#
#     # if scale:
#     #     x /= 255
#     return x


photo_aug_list = [clone, add_additive_shade, add_random_brightness, add_random_contrast, \
                         add_random_saturation, add_additive_shade, add_random_brightness,
                         clone, add_random_contrast, add_random_saturation]

distort_aug_list = [clone, distort_random, distort_random, distort_random, \
                           distort_random, distort_random, distort_random, \
                           distort_random, distort_random, distort_random]

mirror_aug_list = [clone, flip_left_right, flip_left_right, flip_left_right, \
                          flip_left_right, flip_left_right, flip_left_right, \
                          flip_left_right, flip_left_right, flip_left_right]

color_aug_list = [clone, transform_color_space, transform_color_space, transform_color_space, \
                         transform_color_space, transform_color_space, transform_color_space, \
                         transform_color_space, transform_color_space, transform_color_space]

trans_aug_list = [clone, shear_top_down_random, shear_left_right_random, shear_down_top_random, \
                         shear_right_left_random, skew_left_right_random, skew_top_down_random, \
                         skew_top_left_random, skew_left_top_random, skew_down_left, shear_left_down, shear_top_right ]

pres_aug_list =  [clone, shift_random, shift_random, shift_random, \
                                         shift_random, shift_random, shift_random, \
                                         shift_random, shift_random, shift_random]


augmentation_functions = list(set([ tuple(set(fn)) for fn in list(itertools.product(photo_aug_list,
                                                distort_aug_list,
                                                mirror_aug_list,
                                                pres_aug_list + trans_aug_list,
                                                color_aug_list
                                                ))]))

nature_augmentation_functions = list(set([ tuple(set(fn)) for fn in list(itertools.product(photo_aug_list,
                                                distort_aug_list,
                                                mirror_aug_list,
                                                pres_aug_list, trans_aug_list,
                                                color_aug_list
                                                ))]))

augmentation3d_functions = list(set([ tuple(set(fn)) for fn in list(itertools.product(
                                                photo_aug_list, color_aug_list,
                                                distort_aug_list,
                                                mirror_aug_list,
                                                pres_aug_list + trans_aug_list
                                                ))]))
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
