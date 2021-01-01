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



class Augmentor:
    def __init__(self):
        self.augmentation_functions = prepare_functions_list()
        self.last_ix = 0


    def augment(self, images, td_prob=0.25, scale=255.0, batch_shape=None):
        functions_list = self.augmentation_functions[self.last_ix]
        aug_func_name = str([f.__name__ for f in functions_list])
        if np.random.choice([False, True], p=[1 - td_prob, td_prob]):
            td_scales = [a / 100 for a in range(80, 121)]
            done = True
            while done:
                fn = random.choice(functions_list)
                if fn in photo_aug_list:
                    continue
                else:
                    done = False
            aug_patch_fn = lambda images, batch_shape: td_pres_aug.aug_bg_patches(images, td_scales, fn, batch_shape)
            functions_list = [aug_patch_fn if f == fn else f for f in functions_list]
            aug_func_name = aug_func_name.replace(fn.__name__, f"3d{fn.__name__}")
            aug_func_name = aug_func_name.replace('[', '').replace(']', '').replace(',', '').replace(' ', '_').replace("'",
                                                                                                                       '')

        for f in functions_list:
            images = f(images=images, batch_shape=batch_shape)

        self.last_ix += 1
        self.last_ix = self.last_ix % len(self.augmentation_functions)

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
    scales = [a / 1000 for a in range(80, 121)]
    r = random.choice(scales)
    images = pres_aug.rand_shift(images=images, batch_shape=batch_shape, ratio=r)
    return images


def shear_left_right_random(images, batch_shape=None):
    scales = [a / 1000 for a in range(80, 121)]
    l = random.choice(scales)
    images = trans_aug.shear_left_right(images=images, batch_shape=batch_shape, shear_lambda=l)
    return images


def translate_top_down_random(images, batch_shape=None):
    scales = [a / 1000 for a in range(80, 121)]
    l = random.choice(scales)
    images = trans_aug.shear_top_down(images=images, batch_shape=batch_shape, shear_lambda=l)
    return images


def shear_right_left_random(images, batch_shape=None):
    scales = [a / 1000 for a in range(80, 121)]
    l = random.choice(scales)
    images = trans_aug.shear_left_right(images=images, batch_shape=batch_shape, shear_lambda=-l)
    return images

def shear_down_top_random(images, batch_shape=None):
    scales = [a / 1000 for a in range(80, 121)]
    l = random.choice(scales)
    images = trans_aug.shear_down_top(images=images, batch_shape=batch_shape, shear_lambda=-l)
    return images

def shear_top_down_random(images, batch_shape=None):
    scales = [a / 1000 for a in range(80, 121)]
    l = random.choice(scales)
    images = trans_aug.shear_top_down(images=images, batch_shape=batch_shape, shear_lambda=-l)
    return images

def skew_left_right_random(images, batch_shape=None):
    scales = [a / 1000 for a in range(80, 301)]
    d_scales = [a / 100 for a in range(150, 301)]
    ll = random.choice(scales)/random.choice(d_scales)
    lr = random.choice(scales)/random.choice(d_scales)
    images = trans_aug.skew_left_right(images=images, batch_shape=batch_shape, \
                                       l_shear_lambda=ll, r_shear_lambda=lr)
    return images


def skew_top_left_random(images, batch_shape=None):
    scales = [a / 1000 for a in range(80, 301)]
    d_scales = [a / 100 for a in range(150, 301)]
    ll = random.choice(scales)/random.choice(d_scales)
    lt = random.choice(scales)/random.choice(d_scales)
    images = trans_aug.skew_top_left(images=images, batch_shape=batch_shape, \
                                     t_shear_lambda=lt, l_shear_lambda=ll)
    return images

def skew_down_left(images, batch_shape=None):
    scales = [a / 1000 for a in range(80, 301)]
    d_scales = [a / 100 for a in range(150, 301)]
    lt = random.choice(scales)/random.choice(d_scales)
    ll = random.choice(scales)/random.choice(d_scales)
    lt *=  -1
    images = trans_aug.skew_top_left(images=images, batch_shape=batch_shape, \
                                     t_shear_lambda=lt, \
                                     l_shear_lambda=ll)
    return images


def shear_top_right(images, batch_shape=None):
    scales = [a / 1000 for a in range(80, 301)]
    d_scales = [a / 100 for a in range(150, 301)]
    lt = random.choice(scales)/random.choice(d_scales)
    ll = random.choice(scales)/random.choice(d_scales)
    lt  *= -1
    ll *= -1
    images = trans_aug.shear_left_down(images=images, batch_shape=batch_shape, \
                                       t_shear_lambda=lt, \
                                       l_shear_lambda=ll)
    return images


def shear_left_down(images, batch_shape=None):
    scales = [a / 1000 for a in range(80, 301)]
    d_scales = [a / 100 for a in range(150, 301)]
    lt = random.choice(scales)/random.choice(d_scales)
    ll = random.choice(scales)/random.choice(d_scales)
    images = trans_aug.shear_left_down(images=images, batch_shape=batch_shape, \
                                       t_shear_lambda=lt, \
                                       l_shear_lambda=ll)
    return images


def skew_left_top_random(images, batch_shape=None):
    scales = [a / 1000 for a in range(80, 301)]
    d_scales = [a / 100 for a in range(150, 301)]
    lt = random.choice(scales)/random.choice(d_scales)
    ll = random.choice(scales)/random.choice(d_scales)
    images = trans_aug.skew_left_top(images=images, batch_shape=batch_shape, \
                                     t_shear_lambda=lt, \
                                     l_shear_lambda=ll)
    return images


def skew_top_down_random(images, batch_shape=None):
    scales = [a / 1000 for a in range(80, 301)]
    d_scales = [a / 100 for a in range(150, 301)]
    lr = random.choice(scales)/random.choice(d_scales)
    ll = random.choice(scales)/random.choice(d_scales)
    images = trans_aug.skew_top_down(images=images, batch_shape=batch_shape, \
                                     t_shear_lambda=ll, \
                                     d_shear_lambda=lr)

    return images


photo_aug_list = [clone, add_random_contrast, add_random_contrast, add_random_brightness, \
                         add_random_contrast, add_random_contrast, add_random_brightness,
                         add_random_contrast, add_random_contrast, add_random_brightness]

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

all_list = photo_aug_list + distort_aug_list+ mirror_aug_list+ \
                                 color_aug_list+ trans_aug_list+ pres_aug_list

all_fns_list = \
[ tuple(set(fns)) for fns in list(itertools.combinations_with_replacement(all_list, 1)) ] + \
[ tuple(set(fns)) for fns in list(itertools.combinations_with_replacement(all_list, 2)) ] + \
[ tuple(set(fns)) for fns in list(itertools.combinations_with_replacement(all_list, 3)) ] + \
[ tuple(set(fns)) for fns in list(itertools.combinations_with_replacement(all_list, 4)) ] + \
[ tuple(set(fns)) for fns in list(itertools.combinations_with_replacement(all_list, 5)) ]

def prepare_functions_list():
    augmentation_functions = []
    for fns in all_fns_list:
        temp_list = []

        photo_aug_found = False
        distort_aug_found = False
        mirror_aug_found = False
        color_aug_found = False
        trans_aug_found = False
        pres_aug_found = False

        for f in fns:
            if f in photo_aug_list:
                if not photo_aug_found:
                    temp_list += [f]
                    photo_aug_found = True

            elif f in distort_aug_list:
                if not distort_aug_found:
                    temp_list += [f]
                    distort_aug_found = True

            elif f in mirror_aug_list:
                if not mirror_aug_found:
                    temp_list += [f]
                    mirror_aug_found = True

            elif f in color_aug_list:
                if not color_aug_found:
                    temp_list += [f]
                    color_aug_found = True

            elif f in trans_aug_list:
                if not trans_aug_found:
                    temp_list += [f]
                    trans_aug_found = True

            elif f in pres_aug_list:
                if not pres_aug_found:
                    temp_list += [f]
                    pres_aug_found = True

            else:
                pass

        if temp_list != []:
            augmentation_functions += [temp_list]

    return augmentation_functions
