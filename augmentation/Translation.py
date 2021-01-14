
import tensorflow_addons as tfa
import tensorflow as tf
from math import floor, ceil
import random
import numpy as np


def shear_left_right(images, shear_lambda, batch_shape=None):
    if batch_shape is not None:
        _, src_height, src_width,_ = batch_shape
    else:
        im_shape = tf.shape(images)
        src_height, src_width = tf.unstack(im_shape)[1:3]

    images = tf.pad(images, [[0, 0], [5, 5], [5, 5], [0, 0]], 'REFLECT')
    images = tf.image.resize(images, (src_height, src_width))

    pad_size = tf.cast(
        tf.cast(tf.maximum(src_height, src_width), tf.float32) * (2.0 - 1.0) / 2 + 0.5, tf.int32)  # larger than usual (sqrt(2))
    images = tf.pad(images, [[0, 0], [pad_size] * 2, [pad_size] * 2, [0, 0]], 'REFLECT')
    images = tf.pad(images, [[0, 0], [pad_size] * 2, [pad_size] * 2, [0, 0]], 'REFLECT')

    images = transformImg(images, [[1.0, shear_lambda, 0], [0, 1.0, 0], [0, 0, 1.0]])
    return tf.slice(images, [0, pad_size*2, pad_size*2, 0], [-1, src_height, src_width, -1])


def shear_left_down(images, l_shear_lambda, t_shear_lambda, batch_shape=None):
    if batch_shape is not None:
        _, src_height, src_width,_ = batch_shape
    else:
        im_shape = tf.shape(images)
        src_height, src_width = tf.unstack(im_shape)[1:3]

    images = tf.pad(images, [[0, 0], [5, 5], [5, 5], [0, 0]], 'REFLECT')
    images = tf.image.resize(images, (src_height, src_width))

    pad_size = tf.cast(
        tf.cast(tf.maximum(src_height, src_width), tf.float32) * (2.0 - 1.0) / 2 + 0.5,
        tf.int32)  # larger than usual (sqrt(2))
    images = tf.pad(images, [[0, 0], [pad_size] * 2, [pad_size] * 2, [0, 0]], 'REFLECT')
    images = tf.pad(images, [[0, 0], [pad_size] * 2, [pad_size] * 2, [0, 0]], 'REFLECT')
    images = transformImg(images, [[1.0, l_shear_lambda + t_shear_lambda, 0], [t_shear_lambda, 1.0, 0], [0, 0, 1.0]])
    return tf.slice(images, [0, pad_size * 2, pad_size * 2, 0], [-1, src_height, src_width, -1])


def skew_left_right(images, l_shear_lambda, r_shear_lambda, batch_shape=None):
    if batch_shape is not None:
        _, src_height, src_width,_ = batch_shape
    else:
        im_shape = tf.shape(images)
        src_height, src_width = tf.unstack(im_shape)[1:3]

    images = tf.pad(images, [[0, 0], [5, 5], [5, 5], [0, 0]], 'REFLECT')
    images = tf.image.resize(images, (src_height, src_width))

    pad_size = tf.cast(
        tf.cast(tf.maximum(src_height, src_width), tf.float32) * (2.0 - 1.0) / 2 + 0.5,
        tf.int32)  # larger than usual (sqrt(2))
    images = tf.pad(images, [[0, 0], [pad_size] * 2, [pad_size] * 2, [0, 0]], 'REFLECT')
    images = tf.pad(images, [[0, 0], [pad_size] * 2, [pad_size] * 2, [0, 0]], 'REFLECT')
    images = transformImg(images, [[1.0, r_shear_lambda + l_shear_lambda, 0], [0, 1.0, 0], [0, 0, 1.0]])
    images = tf.slice(images, [0, pad_size * 2, pad_size * 2, 0], [-1, src_height, src_width, -1])

    images = tf.image.flip_left_right(images)
    pad_size = tf.cast(
        tf.cast(tf.maximum(src_height, src_width), tf.float32) * (2.0 - 1.0) / 2 + 0.5,
        tf.int32)  # larger than usual (sqrt(2))
    images = tf.pad(images, [[0, 0], [pad_size] * 2, [pad_size] * 2, [0, 0]], 'REFLECT')
    images = tf.pad(images, [[0, 0], [pad_size] * 2, [pad_size] * 2, [0, 0]], 'REFLECT')
    images = transformImg(images, [[1.0, r_shear_lambda, 0], [0, 1.0, 0], [0, 0, 1.0]])
    images = tf.slice(images, [0, pad_size * 2, pad_size * 2, 0], [-1, src_height, src_width, -1])
    return tf.image.flip_left_right(images)


def skew_top_down(images, t_shear_lambda, d_shear_lambda, batch_shape=None):
    images = rot90(images)
    if batch_shape is not None:
        _, src_height, src_width,_ = batch_shape
    else:
        im_shape = tf.shape(images)
        src_height, src_width = tf.unstack(im_shape)[1:3]

    images = tf.pad(images, [[0, 0], [5, 5], [5, 5], [0, 0]], 'REFLECT')
    images = tf.image.resize(images, (src_height, src_width))

    pad_size = tf.cast(
        tf.cast(tf.maximum(src_height, src_width), tf.float32) * (2.0 - 1.0) / 2 + 0.5,
        tf.int32)  # larger than usual (sqrt(2))
    images = tf.pad(images, [[0, 0], [pad_size] * 2, [pad_size] * 2, [0, 0]], 'REFLECT')
    images = tf.pad(images, [[0, 0], [pad_size] * 2, [pad_size] * 2, [0, 0]], 'REFLECT')
    images = transformImg(images, [[1.0, t_shear_lambda + d_shear_lambda, 0], [0, 1.0, 0], [0, 0, 1.0]])
    images = tf.slice(images, [0, pad_size * 2, pad_size * 2, 0], [-1, src_height, src_width, -1])

    images = tf.image.flip_left_right(images)
    pad_size = tf.cast(
        tf.cast(tf.maximum(src_height, src_width), tf.float32) * (2.0 - 1.0) / 2 + 0.5,
        tf.int32)  # larger than usual (sqrt(2))
    images = tf.pad(images, [[0, 0], [pad_size] * 2, [pad_size] * 2, [0, 0]], 'REFLECT')
    images = tf.pad(images, [[0, 0], [pad_size] * 2, [pad_size] * 2, [0, 0]], 'REFLECT')
    images = transformImg(images, [[1.0, t_shear_lambda, 0], [0, 1.0, 0], [0, 0, 1.0]])
    images = tf.slice(images, [0, pad_size * 2, pad_size * 2, 0], [-1, src_height, src_width, -1])
    return tf.image.rot90(tf.image.rot90(tf.image.rot90(tf.image.flip_left_right(images))))


def skew_top_left(images, t_shear_lambda, l_shear_lambda, batch_shape=None):
    if batch_shape is not None:
        _, src_height, src_width,_ = batch_shape
    else:
        im_shape = tf.shape(images)
        src_height, src_width = tf.unstack(im_shape)[1:3]

    images = tf.pad(images, [[0, 0], [5, 5], [5, 5], [0, 0]], 'REFLECT')
    images = tf.image.resize(images, (src_height, src_width))

    pad_size = tf.cast(
        tf.cast(tf.maximum(src_height, src_width), tf.float32) * (2.0 - 1.0) / 2 + 0.5,
        tf.int32)  # larger than usual (sqrt(2))
    images = tf.pad(images, [[0, 0], [pad_size] * 2, [pad_size] * 2, [0, 0]], 'REFLECT')
    images = tf.pad(images, [[0, 0], [pad_size] * 2, [pad_size] * 2, [0, 0]], 'REFLECT')
    images = transformImg(images, [[1.0, t_shear_lambda + l_shear_lambda, 0], [0, 1.0, 0], [0, 0, 1.0]])
    images = tf.slice(images, [0, pad_size * 2, pad_size * 2, 0], [-1, src_height, src_width, -1])

    images = rot90(images)
    pad_size = tf.cast(
        tf.cast(tf.maximum(src_height, src_width), tf.float32) * (2.0 - 1.0) / 2 + 0.5,
        tf.int32)  # larger than usual (sqrt(2))
    images = tf.pad(images, [[0, 0], [pad_size] * 2, [pad_size] * 2, [0, 0]], 'REFLECT')
    images = tf.pad(images, [[0, 0], [pad_size] * 2, [pad_size] * 2, [0, 0]], 'REFLECT')
    images = transformImg(images, [[1.0, t_shear_lambda, 0], [0, 1.0, 0], [0, 0, 1.0]])
    images = tf.slice(images, [0, pad_size * 2, pad_size * 2, 0], [-1, src_height, src_width, -1])
    return tf.image.rot90(tf.image.rot90(rot90(images)))


def skew_left_top(images, t_shear_lambda, l_shear_lambda, batch_shape=None):
    images = rot90(images)
    if batch_shape is not None:
        _, src_height, src_width,_ = batch_shape
    else:
        im_shape = tf.shape(images)
        src_height, src_width = tf.unstack(im_shape)[1:3]

    images = tf.pad(images, [[0, 0], [5, 5], [5, 5], [0, 0]], 'REFLECT')
    images = tf.image.resize(images, (src_height, src_width))

    pad_size = tf.cast(
        tf.cast(tf.maximum(src_height, src_width), tf.float32) * (2.0 - 1.0) / 2 + 0.5,
        tf.int32)  # larger than usual (sqrt(2))
    images = tf.pad(images, [[0, 0], [pad_size] * 2, [pad_size] * 2, [0, 0]], 'REFLECT')
    images = tf.pad(images, [[0, 0], [pad_size] * 2, [pad_size] * 2, [0, 0]], 'REFLECT')
    images = transformImg(images, [[1.0, t_shear_lambda + l_shear_lambda, 0], [0, 1.0, 0], [0, 0, 1.0]])
    images = tf.slice(images, [0, pad_size * 2, pad_size * 2, 0], [-1, src_height, src_width, -1])
    images = tf.image.rot90(tf.image.rot90(rot90(images)))

    pad_size = tf.cast(
        tf.cast(tf.maximum(src_height, src_width), tf.float32) * (2.0 - 1.0) / 2 + 0.5,
        tf.int32)  # larger than usual (sqrt(2))
    images = tf.pad(images, [[0, 0], [pad_size] * 2, [pad_size] * 2, [0, 0]], 'REFLECT')
    images = tf.pad(images, [[0, 0], [pad_size] * 2, [pad_size] * 2, [0, 0]], 'REFLECT')
    images = transformImg(images, [[1.0, t_shear_lambda, 0], [0, 1.0, 0], [0, 0, 1.0]])
    return tf.slice(images, [0, pad_size * 2, pad_size * 2, 0], [-1, src_height, src_width, -1])


def shear_top_down(images, shear_lambda, batch_shape=None):
    images = rot90(images)
    if batch_shape is not None:
        _, src_height, src_width,_ = batch_shape
    else:
        im_shape = tf.shape(images)
        src_height, src_width = tf.unstack(im_shape)[1:3]

    images = tf.pad(images, [[0, 0], [5, 5], [5, 5], [0, 0]], 'REFLECT')
    images = tf.image.resize(images, (src_height, src_width))

    pad_size = tf.cast(
        tf.cast(tf.maximum(src_height, src_width), tf.float32) * (2.0 - 1.0) / 2 + 0.5, tf.int32)  # larger than usual (sqrt(2))
    images = tf.pad(images, [[0, 0], [pad_size] * 2, [pad_size] * 2, [0, 0]], 'REFLECT')
    images = tf.pad(images, [[0, 0], [pad_size] * 2, [pad_size] * 2, [0, 0]], 'REFLECT')

    images = transformImg(images, [[1.0, shear_lambda, 0], [0, 1.0, 0], [0, 0, 1.0]])
    images = tf.slice(images, [0, pad_size*2, pad_size*2, 0], [-1, src_height, src_width, -1])
    return tf.image.rot90(tf.image.rot90(rot90(images)))


def shear_down_top(images, shear_lambda, batch_shape=None):
    images = tf.image.flip_up_down(images)
    images = rot90(images)
    if batch_shape is not None:
        _, src_height, src_width,_ = batch_shape
    else:
        im_shape = tf.shape(images)
        src_height, src_width = tf.unstack(im_shape)[1:3]

    images = tf.pad(images, [[0, 0], [5, 5], [5, 5], [0, 0]], 'REFLECT')
    images = tf.image.resize(images, (src_height, src_width))

    pad_size = tf.cast(
        tf.cast(tf.maximum(src_height, src_width), tf.float32) * (2.0 - 1.0) / 2 + 0.5, tf.int32)  # larger than usual (sqrt(2))
    images = tf.pad(images, [[0, 0], [pad_size] * 2, [pad_size] * 2, [0, 0]], 'REFLECT')
    images = tf.pad(images, [[0, 0], [pad_size] * 2, [pad_size] * 2, [0, 0]], 'REFLECT')

    images = transformImg(images, [[1.0, shear_lambda, 0], [0, 1.0, 0], [0, 0, 1.0]])
    images = tf.slice(images, [0, pad_size*2, pad_size*2, 0], [-1, src_height, src_width, -1])
    return tf.image.flip_up_down(tf.image.rot90(tf.image.rot90(rot90(images))))

def transformImg(imgIn,forward_transform):
    t = tfa.image.transform_ops.matrices_to_flat_transforms(tf.linalg.inv(forward_transform))
    return tfa.image.transform(imgIn, t, interpolation="BILINEAR")


def skew_random_1(images, batch_shape=None):
    if batch_shape is not None:
        _, src_height, src_width,_ = batch_shape
    else:
        im_shape = tf.shape(images)
        src_height, src_width = tf.unstack(im_shape)[1:3]

    case_true = tf.image.resize(tf.slice(images, [0, 0, 0, 0], [-1, src_height, 10, -1], name=None), (src_height, src_width))

    skew_matrix = get_skew_matrix(src_height, src_width, skew_type="RANDOM", magnitude=1)
    images = tf.image.resize(tfa.image.transform(images, skew_matrix, interpolation="BILINEAR"), (src_height, src_width))
    condition = tf.equal(images, 0)
    images = tf.where(condition, case_true, images)
    return images


def skew_random_2(images, batch_shape=None):
    if batch_shape is not None:
        _, src_height, src_width,_ = batch_shape
    else:
        im_shape = tf.shape(images)
        src_height, src_width = tf.unstack(im_shape)[1:3]

    images = tf.image.flip_up_down(images)
    images = rot90(images)

    case_true = tf.image.resize(tf.slice(images, [0, 0, 0, 0], [-1, src_height, 10, -1], name=None), (src_height, src_width))

    skew_matrix = get_skew_matrix(src_height, src_width, skew_type="RANDOM", magnitude=1)
    images = tf.image.resize(tfa.image.transform(images, skew_matrix, interpolation="BILINEAR"), (src_height, src_width))
    condition = tf.equal(images, 0)
    images = tf.where(condition, case_true, images)
    return tf.image.flip_up_down(tf.image.rot90(tf.image.rot90(rot90(images))))


def skew_random_3(images, batch_shape=None):
    if batch_shape is not None:
        _, src_height, src_width,_ = batch_shape
    else:
        im_shape = tf.shape(images)
        src_height, src_width = tf.unstack(im_shape)[1:3]

    images = rot90(images)

    case_true = tf.image.resize(tf.slice(images, [0, 0, 0, 0], [-1, src_height, 10, -1], name=None), (src_height, src_width))

    skew_matrix = get_skew_matrix(src_height, src_width, skew_type="RANDOM", magnitude=1)
    images = tf.image.resize(tfa.image.transform(images, skew_matrix, interpolation="BILINEAR"), (src_height, src_width))
    condition = tf.equal(images, 0)
    images = tf.where(condition, case_true, images)
    return tf.image.rot90(tf.image.rot90(rot90(images)))


def skew_random_4(images, batch_shape=None):
    if batch_shape is not None:
        _, src_height, src_width,_ = batch_shape
    else:
        im_shape = tf.shape(images)
        src_height, src_width = tf.unstack(im_shape)[1:3]

    images = rot90(images)
    images = tf.image.flip_up_down(images)


    case_true = tf.image.resize(tf.slice(images, [0, 0, 0, 0], [-1, src_height, 10, -1], name=None), (src_height, src_width))

    skew_matrix = get_skew_matrix(src_height, src_width, skew_type="RANDOM", magnitude=1)
    images = tf.image.resize(tfa.image.transform(images, skew_matrix, interpolation="BILINEAR"), (src_height, src_width))
    condition = tf.equal(images, 0)
    images = tf.where(condition, case_true, images)
    return tf.image.rot90(tf.image.rot90(rot90(tf.image.flip_up_down(images))))



@tf.function
def rot90(images):
    return tf.transpose(tf.reverse(images, [2]), [0, 2, 1, 3])


def get_skew_matrix(w, h, skew_type="RANDOM", magnitude=10):
    """
    Perform the skew on the passed image(s) and returns the transformed
    image(s). Uses the :attr:`skew_type` and :attr:`magnitude` parameters
    to control the type of skew to perform as well as the degree to which
    it is performed.
    If a list of images is passed, they must have identical dimensions.
    This is checked when we add the ground truth directory using
    :func:`Pipeline.:func:`~Augmentor.Pipeline.Pipeline.ground_truth`
    function.
    However, if this check fails, the skew function will be skipped and
    a warning thrown, in order to avoid an exception.
    :param images: The image(s) to skew.
    :type images: List containing PIL.Image object(s).
    :return: The transformed image(s) as a list of object(s) of type
     PIL.Image.
    """

    # Width and height taken from first image in list.
    # This requires that all ground truth images in the list
    # have identical dimensions!
    # w, h = images[0].size

    x1 = 0
    x2 = h
    y1 = 0
    y2 = w

    original_plane = [(y1, x1), (y2, x1), (y2, x2), (y1, x2)]

    max_skew_amount = max(w, h) // 5
    max_skew_amount = int(ceil(max_skew_amount * magnitude))
    skew_amount = random.randint(1, max_skew_amount)

    # Old implementation, remove.
    # if not self.magnitude:
    #    skew_amount = random.randint(1, max_skew_amount)
    # elif self.magnitude:
    #    max_skew_amount /= self.magnitude
    #    skew_amount = max_skew_amount

    if skew_type == "RANDOM":
        skew = random.choice(["TILT", "TILT_LEFT_RIGHT", "TILT_TOP_BOTTOM", "CORNER"])
    else:
        skew = skew_type

    # We have two choices now: we tilt in one of four directions
    # or we skew a corner.

    if skew == "TILT" or skew == "TILT_LEFT_RIGHT" or skew == "TILT_TOP_BOTTOM":

        if skew == "TILT":
            skew_direction = random.randint(0, 3)
        elif skew == "TILT_LEFT_RIGHT":
            skew_direction = random.randint(0, 1)
        elif skew == "TILT_TOP_BOTTOM":
            skew_direction = random.randint(2, 3)

        if skew_direction == 0:
            # Left Tilt
            new_plane = [(y1, x1 - skew_amount),  # Top Left
                         (y2, x1),  # Top Right
                         (y2, x2),  # Bottom Right
                         (y1, x2 + skew_amount)]  # Bottom Left
        elif skew_direction == 1:
            # Right Tilt
            new_plane = [(y1, x1),  # Top Left
                         (y2, x1 - skew_amount),  # Top Right
                         (y2, x2 + skew_amount),  # Bottom Right
                         (y1, x2)]  # Bottom Left
        elif skew_direction == 2:
            # Forward Tilt
            new_plane = [(y1 - skew_amount, x1),  # Top Left
                         (y2 + skew_amount, x1),  # Top Right
                         (y2, x2),  # Bottom Right
                         (y1, x2)]  # Bottom Left
        elif skew_direction == 3:
            # Backward Tilt
            new_plane = [(y1, x1),  # Top Left
                         (y2, x1),  # Top Right
                         (y2 + skew_amount, x2),  # Bottom Right
                         (y1 - skew_amount, x2)]  # Bottom Left

    if skew == "CORNER":

        skew_direction = random.randint(0, 7)

        if skew_direction == 0:
            # Skew possibility 0
            new_plane = [(y1 - skew_amount, x1), (y2, x1), (y2, x2), (y1, x2)]
        elif skew_direction == 1:
            # Skew possibility 1
            new_plane = [(y1, x1 - skew_amount), (y2, x1), (y2, x2), (y1, x2)]
        elif skew_direction == 2:
            # Skew possibility 2
            new_plane = [(y1, x1), (y2 + skew_amount, x1), (y2, x2), (y1, x2)]
        elif skew_direction == 3:
            # Skew possibility 3
            new_plane = [(y1, x1), (y2, x1 - skew_amount), (y2, x2), (y1, x2)]
        elif skew_direction == 4:
            # Skew possibility 4
            new_plane = [(y1, x1), (y2, x1), (y2 + skew_amount, x2), (y1, x2)]
        elif skew_direction == 5:
            # Skew possibility 5
            new_plane = [(y1, x1), (y2, x1), (y2, x2 + skew_amount), (y1, x2)]
        elif skew_direction == 6:
            # Skew possibility 6
            new_plane = [(y1, x1), (y2, x1), (y2, x2), (y1 - skew_amount, x2)]
        elif skew_direction == 7:
            # Skew possibility 7
            new_plane = [(y1, x1), (y2, x1), (y2, x2), (y1, x2 + skew_amount)]

    if skew_type == "ALL":
        # Not currently in use, as it makes little sense to skew by the same amount
        # in every direction if we have set magnitude manually.
        # It may make sense to keep this, if we ensure the skew_amount below is randomised
        # and cannot be manually set by the user.
        corners = dict()
        corners["top_left"] = (y1 - random.randint(1, skew_amount), x1 - random.randint(1, skew_amount))
        corners["top_right"] = (y2 + random.randint(1, skew_amount), x1 - random.randint(1, skew_amount))
        corners["bottom_right"] = (y2 + random.randint(1, skew_amount), x2 + random.randint(1, skew_amount))
        corners["bottom_left"] = (y1 - random.randint(1, skew_amount), x2 + random.randint(1, skew_amount))

        new_plane = [corners["top_left"], corners["top_right"], corners["bottom_right"], corners["bottom_left"]]

    # To calculate the coefficients required by PIL for the perspective skew,
    # see the following Stack Overflow discussion: https://goo.gl/sSgJdj
    matrix = []

    for p1, p2 in zip(new_plane, original_plane):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])

    A = np.matrix(matrix, dtype=np.float)
    B = np.array(original_plane).reshape(8)

    perspective_skew_coefficients_matrix = np.dot(np.linalg.pinv(A), B)
    perspective_skew_coefficients_matrix = np.array(perspective_skew_coefficients_matrix).reshape(8)
    return perspective_skew_coefficients_matrix


