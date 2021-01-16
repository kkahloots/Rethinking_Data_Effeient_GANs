import random
import itertools
import tensorflow as tf
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

    def augment(self, images, batch_shape, scale=255.0,  print_fn=False, functions_list=None):
        if functions_list is None:
            func_keys = random.sample([*self.augmentation_functions.keys()], random.randint(1, 3))

            functions_list = [random.sample(self.augmentation_functions[k], 1)[0] for k in func_keys]
            functions_list = list(map(lambda f: f(batch_shape) ,functions_list))
            #
            #
            # td_prob = 0.1
            # if np.random.choice([False, True], p=[1 - td_prob, td_prob]):
            #     fn, kw = random.choice(functions_list)
            #     print('3d', fn.__name__)
            #
            #     def aug_patch_fn(batch_shape):
            #         td_scales = [a / 100 for a in range(80, 121)]
            #         kwargs = {'fn': fn, 'scale': td_scales, 'kwargs': kw}
            #         return td_pres_aug.aug_bg_patches, kwargs
            #
            #     functions_list = [aug_patch_fn(batch_shape) if f == fn else (f, kw)  for f,kw in functions_list]

        if print_fn:
            aug_func_name = str([f.__name__ for f, kw in functions_list])
            print(aug_func_name)

        for (f, kw) in functions_list:
            images = call_fn(f, images, kw)

        return images/scale, functions_list


def call_fn(fn, images, kwargs):
    return fn(images, **kwargs)


def clone(batch_shape):
    kwargs = {}
    def c(images, **kw):
        return images
    return c, kwargs


def brightness_random(batch_shape):
    batch_size, width, height, ch = batch_shape
    kwargs = {
       'magnitude': tf.random.uniform([batch_size, 1, 1, 1], minval=-100, maxval=100)
    }
    return photo_aug.random_brightness, kwargs


def contrast_random(batch_shape):
    batch_size, width, height, ch = batch_shape
    kwargs = {
       'magnitude': tf.random.uniform([batch_size, 1, 1, 1], minval=0.5, maxval=1.5)
    }
    return photo_aug.random_contrast, kwargs


def saturation_random(batch_shape):
    batch_size, width, height, ch = batch_shape
    kwargs = {
       'magnitude': tf.random.uniform([batch_size, 1, 1, 1], minval=0.5, maxval=1.5)
    }
    return photo_aug.random_saturation, kwargs


def transform_color_space(batch_shape):
    kwargs = {'flag': random.choice(color_aug.flags)}
    return color_aug.color_space_transform, kwargs


def rotate_random(batch_shape):
    batch_size, width, height, ch = batch_shape
    kwargs = {'width': width,
              'height': height,
              'angles': random.randint(-35, 35)}
    return pres_aug.rotate, kwargs


def flip_left_right(batch_shape):
    kwargs = {}
    return mirror_aug.flip_left_right, kwargs


def distort_random(batch_shape):
    batch_size, width, height, ch = batch_shape
    num_anchors = random.randint(5, 15)
    perturb_sigma = random.randint(-5, 5)
    distortion_x = tf.random.normal((batch_size, num_anchors, num_anchors, 1), stddev=perturb_sigma)
    distortion_y = tf.random.normal((batch_size, num_anchors, num_anchors, 1), stddev=perturb_sigma)
    kwargs = {
        'batch_size': batch_size,
        'height': height,
        'width': width,
        'num_anchors': num_anchors,
        'perturb_sigma': perturb_sigma,
        'distortion_x': distortion_x,
        'distortion_y': distortion_y
    }
    return distort_aug.distort, kwargs


def shift_random(batch_shape):
    batch_size, width, height, ch = batch_shape

    pad_size = tf.cast(
        tf.cast(tf.maximum(height, width), tf.float32) * (2.0 - 1.0) / 2 + 0.5, tf.int32)  # larger than usual (sqrt(2))
    timages = tf.pad(tf.zeros(batch_shape), [[0, 0], [pad_size] * 2, [pad_size] * 2, [0, 0]], 'REFLECT')
    pwidth, pheight = timages.get_shape().as_list()[1:3]

    shift = tf.cast(tf.cast((pwidth, pheight), tf.float32) * random.choice([a / 1000 for a in range(80, 121)]) + 0.5,
                    tf.int32)
    kwargs = {
        'height': height,
        'width': width,
        'pheight': pheight,
        'pwidth': pwidth,
        'translation_x': tf.random.uniform([batch_size, 1], -shift[0], shift[0] + 1, dtype=tf.int32),
        'translation_y': tf.random.uniform([batch_size, 1], -shift[1], shift[1] + 1, dtype=tf.int32)
    }

    return pres_aug.rand_shift, kwargs


def shear_left_random(batch_shape):
    batch_size, width, height, ch = batch_shape
    kwargs = {
        'height': height,
        'width': width,
        'shear_lambda': random.choice([a / 1000 for a in range(80, 121)])
    }

    return trans_aug.shear_left, kwargs


def shear_right_random(batch_shape):
    batch_size, width, height, ch = batch_shape
    kwargs = {
        'height': height,
        'width': width,
        'shear_lambda': random.choice([a / 1000 for a in range(80, 121)])
    }

    return trans_aug.shear_right, kwargs


def shear_left_down_random(batch_shape):
    batch_size, width, height, ch = batch_shape
    kwargs = {
        'height': height,
        'width': width,
        'shear_lambda1': random.choice([a / 1000 for a in range(80, 121)]),
        'shear_lambda2': random.choice([a / 1000 for a in range(80, 121)])
    }

    return trans_aug.shear_left_down, kwargs


def shear_right_down_random(batch_shape):
    batch_size, width, height, ch = batch_shape
    kwargs = {
        'height': height,
        'width': width,
        'shear_lambda1': random.choice([a / 1000 for a in range(80, 121)]),
        'shear_lambda2': random.choice([a / 1000 for a in range(80, 121)])
    }

    return trans_aug.shear_right_down, kwargs


def shear_down_left_random(batch_shape):
    batch_size, width, height, ch = batch_shape
    kwargs = {
        'height': height,
        'width': width,
        'shear_lambda': random.choice([a / 1000 for a in range(80, 121)]),
    }

    return trans_aug.shear_down_left, kwargs


def shear_down_right_random(batch_shape):
    batch_size, width, height, ch = batch_shape
    kwargs = {
        'height': height,
        'width': width,
        'shear_lambda': random.choice([a / 1000 for a in range(80, 121)]),
    }

    return trans_aug.shear_down_right, kwargs


def shear_left_up_random(batch_shape):
    batch_size, width, height, ch = batch_shape
    kwargs = {
        'height': height,
        'width': width,
        'shear_lambda1': random.choice([a / 1000 for a in range(80, 121)]),
        'shear_lambda2': random.choice([a / 1000 for a in range(80, 121)])
    }

    return trans_aug.shear_left_up, kwargs


def shear_right_up_random(batch_shape):
    batch_size, width, height, ch = batch_shape
    kwargs = {
        'height': height,
        'width': width,
        'shear_lambda1': random.choice([a / 1000 for a in range(80, 121)]),
        'shear_lambda2': random.choice([a / 1000 for a in range(80, 121)])
    }

    return trans_aug.shear_right_up, kwargs


#############################
def nshear_left_random(batch_shape):
    batch_size, width, height, ch = batch_shape
    kwargs = {
        'height': height,
        'width': width,
        'shear_lambda': -1 * random.choice([a / 1000 for a in range(80, 121)])
    }

    return trans_aug.shear_left, kwargs


def nshear_right_random(batch_shape):
    batch_size, width, height, ch = batch_shape
    kwargs = {
        'height': height,
        'width': width,
        'shear_lambda': -1 * random.choice([a / 1000 for a in range(80, 121)])
    }

    return trans_aug.shear_right, kwargs


def nshear_left_down_random(batch_shape):
    batch_size, width, height, ch = batch_shape
    kwargs = {
        'height': height,
        'width': width,
        'shear_lambda1': -1 * random.choice([a / 1000 for a in range(80, 121)]),
        'shear_lambda2': -1 * random.choice([a / 1000 for a in range(80, 121)])
    }

    return trans_aug.shear_left_down, kwargs



def nshear_right_down_random(batch_shape):
    batch_size, width, height, ch = batch_shape
    kwargs = {
        'height': height,
        'width': width,
        'shear_lambda1': -1 * random.choice([a / 1000 for a in range(80, 121)]),
        'shear_lambda2': -1 * random.choice([a / 1000 for a in range(80, 121)])
    }

    return trans_aug.shear_right_down, kwargs


def nshear_down_left_random(batch_shape):
    batch_size, width, height,  ch = batch_shape
    kwargs = {
        'height': height,
        'width': width,
        'shear_lambda': -1 * random.choice([a / 1000 for a in range(80, 121)]),
    }

    return trans_aug.shear_down_left, kwargs


def nshear_down_right_random(batch_shape):
    batch_size, width, height,  ch = batch_shape
    kwargs = {
        'height': height,
        'width': width,
        'shear_lambda': -1 * random.choice([a / 1000 for a in range(80, 121)]),
    }

    return trans_aug.shear_down_right, kwargs


def nshear_left_up_random(batch_shape):
    batch_size, width, height,  ch = batch_shape
    kwargs = {
        'height': height,
        'width': width,
        'shear_lambda1': -1 * random.choice([a / 1000 for a in range(80, 121)]),
        'shear_lambda2': -1 * random.choice([a / 1000 for a in range(80, 121)])
    }

    return trans_aug.shear_left_up, kwargs


#skew_left_right_reflect, skew_left_right_repaint, skew_top_down_reflect, skew_top_down_repaint
##############################
def skew_left_right_reflect_random(batch_shape):
    batch_size, width, height, ch = batch_shape
    kwargs = {
        'height': height,
        'width': width,
        'shear_lambda1': random.choice([a / 1000 for a in range(80, 121)]),
        'shear_lambda2': random.choice([a / 1000 for a in range(80, 121)])
    }

    return trans_aug.skew_left_right_reflect, kwargs


def nskew_left_right_reflect_random(batch_shape):
    batch_size, width, height, ch = batch_shape
    kwargs = {
        'height': height,
        'width': width,
        'shear_lambda1': -1*random.choice([a / 1000 for a in range(80, 121)]),
        'shear_lambda2': -1*random.choice([a / 1000 for a in range(80, 121)])
    }

    return trans_aug.skew_left_right_repaint, kwargs


def skew_left_right_repaint_random(batch_shape):
    batch_size, width, height, ch = batch_shape
    kwargs = {
        'height': height,
        'width': width,
        'shear_lambda1': random.choice([a / 1000 for a in range(80, 121)]),
        'shear_lambda2': random.choice([a / 1000 for a in range(80, 121)])
    }

    return trans_aug.skew_left_right_repaint, kwargs


def nskew_left_right_repaint_random(batch_shape):
    batch_size, width, height, ch = batch_shape
    kwargs = {
        'height': height,
        'width': width,
        'shear_lambda1': -1*random.choice([a / 1000 for a in range(80, 121)]),
        'shear_lambda2': -1*random.choice([a / 1000 for a in range(80, 121)])
    }

    return trans_aug.skew_left_right_reflect, kwargs

##############################
def skew_top_down_reflect_random(batch_shape):
    batch_size, width, height, ch = batch_shape
    kwargs = {
        'height': height,
        'width': width,
        'shear_lambda1': random.choice([a / 1000 for a in range(80, 121)]),
        'shear_lambda2': random.choice([a / 1000 for a in range(80, 121)])
    }

    return trans_aug.skew_top_down_reflect, kwargs


def nskew_top_down_reflect_random(batch_shape):
    batch_size, width, height, ch = batch_shape
    kwargs = {
        'height': height,
        'width': width,
        'shear_lambda1': -1 * random.choice([a / 1000 for a in range(80, 121)]),
        'shear_lambda2': -1 * random.choice([a / 1000 for a in range(80, 121)])
    }

    return trans_aug.skew_top_down_repaint, kwargs


def skew_top_down_repaint_random(batch_shape):
    batch_size, width, height, ch = batch_shape
    kwargs = {
        'height': height,
        'width': width,
        'shear_lambda1': random.choice([a / 1000 for a in range(80, 121)]),
        'shear_lambda2': random.choice([a / 1000 for a in range(80, 121)])
    }

    return trans_aug.skew_top_down_repaint, kwargs


def nskew_top_down_repaint_random(batch_shape):
    batch_size, width, height, ch = batch_shape
    kwargs = {
        'height': height,
        'width': width,
        'shear_lambda1': -1*random.choice([a / 1000 for a in range(80, 121)]),
        'shear_lambda2': -1*random.choice([a / 1000 for a in range(80, 121)])
    }

    return trans_aug.skew_top_down_reflect, kwargs


##############################

def tilt_random(batch_shape):
    batch_size, width, height, ch = batch_shape
    kwargs = {
        'height': height,
        'width': width,
        'skew_matrix': trans_aug.get_skew_matrix(height, width, skew_type="TILT_LEFT_RIGHT", magnitude=random.randint(1, 5))
    }

    return trans_aug.tilt_random, kwargs


def tilt_up_down_random(batch_shape):
    batch_size, width, height, ch = batch_shape
    kwargs = {
        'height': height,
        'width': width,
        'skew_matrix': trans_aug.get_skew_matrix(height, width, skew_type="TILT_LEFT_RIGHT", magnitude=random.randint(1, 5))
    }

    return trans_aug.tilt_up_down_random, kwargs

def tilt_left_random(batch_shape):
    batch_size, width, height, ch = batch_shape
    kwargs = {
        'height': height,
        'width': width,
        'skew_matrix': trans_aug.get_skew_matrix(height, width, skew_type="TILT_LEFT_RIGHT", magnitude=random.randint(1, 5))
    }

    return trans_aug.tilt_left_random, kwargs


def tilt_left_up_down_random(batch_shape):
    batch_size, width, height, ch = batch_shape
    kwargs = {
        'height': height,
        'width': width,
        'skew_matrix': trans_aug.get_skew_matrix(height, width, skew_type="TILT_LEFT_RIGHT", magnitude=random.randint(1, 5))
    }

    return trans_aug.tilt_left_up_down_random, kwargs


def tilt_corner_random(batch_shape):
    batch_size, width, height, ch = batch_shape
    kwargs = {
        'height': height,
        'width': width,
        'skew_matrix': trans_aug.get_skew_matrix(height, width, skew_type="CORNER", magnitude=random.randint(1, 5))
    }

    return trans_aug.tilt_random, kwargs


def tilt_up_down_corner_random(batch_shape):
    batch_size, width, height, ch = batch_shape
    kwargs = {
        'height': height,
        'width': width,
        'skew_matrix': trans_aug.get_skew_matrix(height, width, skew_type="CORNER", magnitude=random.randint(1, 5))
    }

    return trans_aug.tilt_up_down_random, kwargs

def tilt_left_corner_random(batch_shape):
    batch_size, width, height, ch = batch_shape
    kwargs = {
        'height': height,
        'width': width,
        'skew_matrix': trans_aug.get_skew_matrix(height, width, skew_type="CORNER", magnitude=random.randint(1, 5))
    }

    return trans_aug.tilt_left_random, kwargs


def tilt_left_up_down_corner_random(batch_shape):
    batch_size, width, height, ch = batch_shape
    kwargs = {
        'height': height,
        'width': width,
        'skew_matrix': trans_aug.get_skew_matrix(height, width, skew_type="CORNER", magnitude=random.randint(1, 5))
    }

    return trans_aug.tilt_left_up_down_random, kwargs


def cutout_random(batch_shape):
    batch_size, width, height,  ch = batch_shape
    scales = [a / 100 for a in range(10, 51)]
    r = random.choice(scales)
    kwargs = {
    'mask': cutout_aug.rand_mask(batch_size, height, width, ratio=r),
    'width': width,
    'height': height
    }

    return cutout_aug.cutout, kwargs


shear_fns = [shear_left_random, shear_right_random, shear_left_down_random, shear_right_down_random, \
             shear_down_left_random, shear_down_right_random, shear_left_up_random, shear_right_up_random,
             nshear_left_random, nshear_right_random, nshear_left_down_random,
             nshear_right_down_random, nshear_down_left_random, nshear_down_right_random,
             nshear_left_up_random]


skew_fns = [skew_left_right_reflect_random, nskew_left_right_reflect_random,
            skew_left_right_repaint_random, nskew_left_right_repaint_random,
            skew_top_down_reflect_random, nskew_top_down_reflect_random,
            skew_top_down_repaint_random, nskew_top_down_repaint_random]

tilt_fns = [tilt_random, tilt_up_down_random, tilt_left_random, tilt_left_up_down_random,
            tilt_corner_random, tilt_up_down_corner_random, tilt_left_corner_random,
            tilt_left_up_down_corner_random]

AUGMENT_FNS = {
    'clone':   [clone],
    'photo':   [contrast_random, saturation_random, brightness_random],
    'color':   [transform_color_space],
    'cutout':  [cutout_random],
    'distort': [distort_random],
    'mirror':  [flip_left_right],
    'shift':   [shift_random],
    'rotate':  [rotate_random],
    'skew':    shear_fns+skew_fns+tilt_fns
}
