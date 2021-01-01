import random
import itertools

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


def AugmentObject(images, td_prob=0.3, scale=255.0, batch_shape=None):
    aug_3d = False#np.random.choice([False, True], p=[1-td_prob, td_prob])
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
        aug_patch_fn = lambda images, batch_shape: td_pres_aug.aug_bg_patches(images, td_scales, fn, batch_shape)
        functions_list = [aug_patch_fn if f==fn else f for f in functions_list]
    else:
        functions_list = random.choice(augmentation_functions)

    for f in functions_list:
        print(f)
        images = f(images=images, batch_shape=batch_shape)

    return images/scale


def AugmentNature(images, td_prob=0.3, scale=255.0, batch_shape=None):
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
        aug_patch_fn = lambda images, batch_shape: td_pres_aug.aug_bg_patches(images, td_scales, fn, batch_shape)
        functions_list = [aug_patch_fn if f==fn else f for f in functions_list]
    else:
        functions_list = random.choice(nature_augmentation_functions)

    for f in functions_list:
        images = f(images=images, batch_shape=batch_shape)

    return images/scale


def clone(images, batch_shape=None):
    return images


def add_random_brightness(images, batch_shape=None):
    images = photo_aug.random_brightness(images=images, max_abs_change=100, batch_shape=batch_shape)
    return images


def add_random_contrast(images, batch_shape=None):
    images = photo_aug.random_contrast(images=images, batch_shape=batch_shape)
    return images


def add_random_saturation(images, batch_shape=None):
    images = photo_aug.random_saturation(images=images, batch_shape=batch_shape)
    return images


def add_additive_shade(images, batch_shape=None):
    images = photo_aug.additive_shade(images=images, batch_shape=batch_shape)
    return images


def transform_color_space(images, batch_shape=None):
    images = color_aug.color_space_transform(images=images, batch_shape=batch_shape)
    return images


def rotate_random(images, batch_shape=None):
    a = random.randint(-35, 35)
    images = pres_aug.rotate(images=images, batch_shape=batch_shape, angles=a)
    return images


def flip_left_right(images, batch_shape=None):
    images = mirror_aug.flip_left_right(images=images, batch_shape=batch_shape)
    return images


def distort_random(images, batch_shape=None):
    n = random.randint(5, 15)
    s = random.randint(-5, 5)
    images = distort_aug.distort(images=images, batch_shape=batch_shape, num_anchors=n, perturb_sigma=s)

    return images


def shift_random(images, batch_shape=None):
    r = random.choice(scales)
    images = pres_aug.rand_shift(images=images, batch_shape=batch_shape, ratio=r)
    return images


def shear_left_right_random(images, batch_shape=None):
    l = random.choice(scales)
    images = trans_aug.shear_left_right(images=images, batch_shape=batch_shape, shear_lambda=l)
    return images


def translate_top_down_random(images, batch_shape=None):
    l = random.choice(scales)
    images = trans_aug.shear_top_down(images=images, batch_shape=batch_shape, shear_lambda=l)
    return images


def shear_right_left_random(images, batch_shape=None):
    l = random.choice(scales)
    images = trans_aug.shear_left_right(images=images, batch_shape=batch_shape, shear_lambda=-l)
    return images

def shear_down_top_random(images, batch_shape=None):
    l = random.choice(scales)
    images = trans_aug.shear_down_top(images=images, batch_shape=batch_shape, shear_lambda=-l)
    return images

def shear_top_down_random(images, batch_shape=None):
    l = random.choice(scales)
    images = trans_aug.shear_top_down(images=images, batch_shape=batch_shape, shear_lambda=-l)
    return images

def skew_left_right_random(images, batch_shape=None):
    ll = random.choice(scales)
    lr = random.choice(scales)
    images = trans_aug.skew_left_right(images=images, batch_shape=batch_shape, \
                                       l_shear_lambda=ll/random.choice(d_scales), r_shear_lambda=lr/random.choice(d_scales))
    return images


def skew_top_left_random(images, batch_shape=None):
    lt = random.choice(scales)
    ll = random.choice(scales)
    images = trans_aug.skew_top_left(images=images, batch_shape=batch_shape, \
                                     t_shear_lambda=lt/random.choice(d_scales), l_shear_lambda=ll/random.choice(d_scales))
    return images

def skew_down_left(images, batch_shape=None):
    lt = random.choice(scales) * -1
    ll = random.choice(scales)
    images = trans_aug.skew_top_left(images=images, batch_shape=batch_shape, \
                                     t_shear_lambda=lt/random.choice(d_scales), \
                                     l_shear_lambda=ll/random.choice(d_scales))
    return images


def shear_top_right(images, batch_shape=None):
    lt = random.choice(scales) * -1
    ll = random.choice(scales) * -1
    images = trans_aug.shear_left_down(images=images, batch_shape=batch_shape, \
                                       t_shear_lambda=lt/random.choice(d_scales), \
                                       l_shear_lambda=ll/random.choice(d_scales))
    return images


def shear_left_down(images, batch_shape=None):
    lt = random.choice(scales)
    ll = random.choice(scales)
    images = trans_aug.shear_left_down(images=images, batch_shape=batch_shape, \
                                       t_shear_lambda=lt/random.choice(d_scales), \
                                       l_shear_lambda=ll/random.choice(d_scales))
    return images


def skew_left_top_random(images, batch_shape=None):

    lt = random.choice(scales)
    ll = random.choice(scales)
    images = trans_aug.skew_left_top(images=images, batch_shape=batch_shape, \
                                     t_shear_lambda=lt/random.choice(d_scales), \
                                     l_shear_lambda=ll/random.choice(d_scales))
    return images


def skew_top_down_random(images, batch_shape=None):
    ll = random.choice(scales)
    lr = random.choice(scales)
    images = trans_aug.skew_top_down(images=images, batch_shape=batch_shape, \
                                     t_shear_lambda=ll/random.choice(d_scales), \
                                     d_shear_lambda=lr/random.choice(d_scales))

    return images



photo_aug_list = [clone, add_additive_shade, add_random_contrast, add_random_brightness, \
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


augmentation_functions = list(set([ tuple(set(fn)) for fn in list(itertools.product(
                                                photo_aug_list,
                                                distort_aug_list,
                                                mirror_aug_list,
                                                pres_aug_list , trans_aug_list,
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
