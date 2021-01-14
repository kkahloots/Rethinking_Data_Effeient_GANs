import random
import itertools

import augmentation.Coloring as color_aug
import augmentation.Distortion as distort_aug
import augmentation.Mirror as mirror_aug
import augmentation.Cutout as cutout_aug
import augmentation.Perspective as pres_aug
import augmentation.Photometric as photo_aug
import augmentation.Translation as trans_aug
import augmentation.ThreeDimension_Presective as td_pres_aug
import numpy as np



class Augmentor:
    def __init__(self):
        self.augmentation_functions = AUGMENT_FNS


    def augment(self, images, scale=255.0, batch_shape=None, print_fn=False):
        func_keys = random.sample([*self.augmentation_functions.keys()], random.randint(1, 5))

        functions_list = [random.sample(self.augmentation_functions[k],1)[0] for k in func_keys]
        aug_func_name = str([f.__name__ for f in functions_list])

        if print_fn:
            print(aug_func_name)

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
    a = random.randint(-25, 25)
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



def skew_random1(images, batch_shape=None):
    images = trans_aug.skew_random_1(images=images, batch_shape=batch_shape)
    return images

def skew_random2(images, batch_shape=None):
    images = trans_aug.skew_random_2(images=images, batch_shape=batch_shape)
    return images

def skew_random3(images, batch_shape=None):
    images = trans_aug.skew_random_3(images=images, batch_shape=batch_shape)
    return images

def skew_random4(images, batch_shape=None):
    images = trans_aug.skew_random_4(images=images, batch_shape=batch_shape)
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


def cutout_random(images, batch_shape=None):
    scales = [a/100 for a in range(10, 51)]
    r = random.choice(scales)
    images = cutout_aug.cutout(images=images, batch_shape=batch_shape, ratio=r)
    return images


AUGMENT_FNS = {
    'clone' : [clone],
    'photo': [add_random_contrast, add_random_contrast, add_random_brightness],
    'color': [transform_color_space],
    'cutout': [cutout_random],
    'distort': [distort_random],
    'shear': [shear_top_down_random, shear_left_right_random, shear_down_top_random, \
                    shear_right_left_random,  shear_left_down, shear_top_right ],
    'mirror': [flip_left_right],
    'skew': [skew_left_right_random, skew_top_down_random, skew_top_left_random, skew_left_top_random, skew_down_left],
    'skew2':  [skew_random1, skew_random2, skew_random3, skew_random4],
    'shift':  [shift_random]

}


