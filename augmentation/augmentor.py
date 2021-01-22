import random
import tensorflow as tf
import augmentation.Coloring as color_aug
import augmentation.Distortion as distort_aug
import augmentation.Mirror as mirror_aug
import augmentation.Cutout as cutout_aug
import augmentation.Perspective as pres_aug
import augmentation.Photometric as photo_aug
import augmentation.Translation as trans_aug
import numpy as np


class Augmentor:
    def __init__(self):
        self.augmentation_functions = AUGMENT_FNS

    def augment(self, images, batch_shape, scale=255.0,  print_fn=False):
        ix_lists = np.split(np.arange(batch_shape[0]), max(2, batch_shape[0]//6))
        aug_image = []
        for ix_list in ix_lists:
            func_keys = random.sample([*self.augmentation_functions.keys()], random.randint(1, 3))

            functions_list = [random.sample(self.augmentation_functions[k], 1)[0] for k in func_keys]
            functions_list = list(map(lambda f: f([len(ix_list), *batch_shape[1:]]) ,functions_list))

            if print_fn:
                aug_func_name = str([f.__name__ for f, kw in functions_list])
                print(aug_func_name)

            timg = images[ix_list[0]:ix_list[-1]+1]
            for (f, kw) in functions_list:
                timg = call_fn(f, timg, kw)

            aug_image += [timg]

        return tf.random.shuffle(tf.concat(aug_image, axis=0))/scale


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


def shear_rot90_random(batch_shape):
    batch_size, width, height, ch = batch_shape
    kwargs = {
        'height': height,
        'width': width,
        'shear_lambda': random.choice([a / 1000 for a in range(80, 121)])
    }

    return trans_aug.shear_rot90, kwargs


def ishear_left_random(batch_shape):
    batch_size, width, height, ch = batch_shape
    kwargs = {
        'height': height,
        'width': width,
        'shear_lambda': random.choice([a / 1000 for a in range(80, 121)])
    }

    return trans_aug.ishear_left, kwargs


def ishear_right_random(batch_shape):
    batch_size, width, height, ch = batch_shape
    kwargs = {
        'height': height,
        'width': width,
        'shear_lambda': random.choice([a / 1000 for a in range(80, 121)])
    }

    return trans_aug.ishear_right, kwargs


def ishear_rot90_random(batch_shape):
    batch_size, width, height, ch = batch_shape
    kwargs = {
        'height': height,
        'width': width,
        'shear_lambda': random.choice([a / 1000 for a in range(80, 121)])
    }

    return trans_aug.ishear_rot90, kwargs


#####

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


def shear_rot90_down_random(batch_shape):
    batch_size, width, height, ch = batch_shape
    kwargs = {
        'height': height,
        'width': width,
        'shear_lambda1': random.choice([a / 1000 for a in range(80, 121)]),
        'shear_lambda2': random.choice([a / 1000 for a in range(80, 121)])
    }

    return trans_aug.shear_rot90_down, kwargs


def ishear_left_down_random(batch_shape):
    batch_size, width, height, ch = batch_shape
    kwargs = {
        'height': height,
        'width': width,
        'shear_lambda1': random.choice([a / 1000 for a in range(80, 121)]),
        'shear_lambda2': random.choice([a / 1000 for a in range(80, 121)])
    }

    return trans_aug.ishear_left_down, kwargs


def ishear_right_down_random(batch_shape):
    batch_size, width, height, ch = batch_shape
    kwargs = {
        'height': height,
        'width': width,
        'shear_lambda1': random.choice([a / 1000 for a in range(80, 121)]),
        'shear_lambda2': random.choice([a / 1000 for a in range(80, 121)])
    }

    return trans_aug.ishear_right_down, kwargs


def ishear_rot90_down_random(batch_shape):
    batch_size, width, height, ch = batch_shape
    kwargs = {
        'height': height,
        'width': width,
        'shear_lambda1': random.choice([a / 1000 for a in range(80, 121)]),
        'shear_lambda2': random.choice([a / 1000 for a in range(80, 121)])
    }

    return trans_aug.ishear_rot90_down, kwargs


##############################
def tilt_left_random(batch_shape):
    batch_size, width, height, ch = batch_shape
    kwargs = {
        'height': height,
        'width': width,
        'skew_matrix': trans_aug.get_skew_matrix(height, width, skew_type="TILT_LEFT_RIGHT",
                                                 magnitude=random.randint(1, 5))
    }

    return trans_aug.tilt_left_random, kwargs

def tilt_up_random(batch_shape):
    batch_size, width, height, ch = batch_shape
    kwargs = {
        'height': height,
        'width': width,
        'skew_matrix': trans_aug.get_skew_matrix(height, width, skew_type="TILT_LEFT_RIGHT",
                                                 magnitude=random.randint(1, 5))
    }

    return trans_aug.tilt_up_random, kwargs

##############################

def tilt_corner_left_random(batch_shape):
    batch_size, width, height, ch = batch_shape
    kwargs = {
        'height': height,
        'width': width,
        'skew_matrix': trans_aug.get_skew_matrix(height, width, skew_type="CORNER",
                                                 magnitude=random.randint(1, 5))
    }

    return trans_aug.tilt_left_random, kwargs

def tilt_corner_up_random(batch_shape):
    batch_size, width, height, ch = batch_shape
    kwargs = {
        'height': height,
        'width': width,
        'skew_matrix': trans_aug.get_skew_matrix(height, width, skew_type="CORNER",
                                                 magnitude=random.randint(1, 5))
    }

    return trans_aug.tilt_up_random, kwargs

##############################

def cutout_random(batch_shape):
    batch_size, width, height,  ch = batch_shape
    scales = [a / 100 for a in range(10, 26)]
    r = random.choice(scales)
    kwargs = {
    'mask': cutout_aug.rand_mask(batch_size, height, width, ratio=r),
    'width': width,
    'height': height
    }

    return cutout_aug.cutout, kwargs



shear_fns = [ishear_rot90_down_random, ishear_right_down_random, ishear_left_down_random,
             shear_rot90_down_random, shear_right_down_random ,shear_left_down_random,
             ishear_rot90_random, ishear_right_random, ishear_left_random, shear_rot90_random,
             shear_right_random, shear_left_random]


AUGMENT_FNS = {
    'clone':   [clone],
    'shear':   shear_fns,
    'tilt':    [tilt_left_random, tilt_up_random, tilt_corner_up_random, tilt_corner_left_random],
    'photo':   [contrast_random, saturation_random, brightness_random],
    'color':   [transform_color_space],
    'distort': [distort_random],
    'mirror':  [flip_left_right],
    'shift':   [shift_random],
    'rotate':  [rotate_random]
}
#