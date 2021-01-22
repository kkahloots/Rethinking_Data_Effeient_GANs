
import tensorflow_addons as tfa
import tensorflow as tf
from math import floor, ceil
import random
import numpy as np
import cv2
from augmentation.Cutout import inpaint


def shear_left(images, **kwargs):

    images = tf.pad(images, [[0, 0], [5, 5], [5, 5], [0, 0]], 'REFLECT')
    images = tf.image.resize(images, (kwargs['height'], kwargs['width']))

    pad_size = tf.cast(
        tf.cast(tf.maximum(kwargs['height'], kwargs['width']), tf.float32) * (2.0 - 1.0) / 2 + 0.5, tf.int32)  # larger than usual (sqrt(2))
    images = tf.pad(images, [[0, 0], [pad_size] * 2, [pad_size] * 2, [0, 0]], 'REFLECT')
    images = tf.pad(images, [[0, 0], [pad_size] * 2, [pad_size] * 2, [0, 0]], 'REFLECT')

    images = transformImg(images, [[1.0, kwargs['shear_lambda'], 0], [0, 1.0, 0], [0, 0, 1.0]])
    return tf.slice(images, [0, pad_size*2  , pad_size*2 + kwargs['width']//5 , 0], [-1, kwargs['height'], kwargs['width'], -1])


def shear_right(images, **kwargs):
    return tf.image.flip_left_right(
        tf.image.flip_left_right(
            shear_left(images, **kwargs)
        )
    )

def shear_rot90(images, **kwargs):
    return rot90(rot90(rot90(shear_left(rot90(images), **kwargs))))

def ishear_left(images, **kwargs):
    images = tf.image.flip_up_down(images)
    return tf.image.flip_up_down(shear_left(images, **kwargs))

def ishear_right(images, **kwargs):
    images = tf.image.flip_up_down(images)
    return tf.image.flip_up_down(shear_right(images, **kwargs))

def ishear_rot90(images, **kwargs):
    images = tf.image.flip_up_down(images)
    return tf.image.flip_up_down(shear_rot90(images, **kwargs))



############
def shear_left_down(images, **kwargs):

    images = tf.pad(images, [[0, 0], [5, 5], [5, 5], [0, 0]], 'REFLECT')
    images = tf.image.resize(images, (kwargs['height'], kwargs['width']))

    pad_size = tf.cast(
        tf.cast(tf.maximum(kwargs['height'], kwargs['width']), tf.float32) * (2.0 - 1.0) / 2 + 0.5,
        tf.int32)  # larger than usual (sqrt(2))
    images = tf.pad(images, [[0, 0], [pad_size] * 2, [pad_size] * 2, [0, 0]], 'REFLECT')
    images = tf.pad(images, [[0, 0], [pad_size] * 2, [pad_size] * 2, [0, 0]], 'REFLECT')
    images = transformImg(images, [[1.0, kwargs['shear_lambda2'] + kwargs['shear_lambda1'], 0], [kwargs['shear_lambda1'], 1.0, 0], [0, 0, 1.0]])
    return tf.slice(images, [0, pad_size*2  + kwargs['height']//5, pad_size*2  + kwargs['width']//3, 0], [-1, kwargs['height'], kwargs['width'], -1])


def shear_right_down(images, **kwargs):
    return tf.image.flip_left_right(
        tf.image.flip_left_right(
            shear_left_down(images, **kwargs)
        )
    )

def shear_rot90_down(images, **kwargs):
    return rot90(rot90(rot90(shear_left_down(rot90(images), **kwargs))))

def ishear_left_down(images, **kwargs):
    images = tf.image.flip_up_down(images)
    return tf.image.flip_up_down(shear_left_down(images, **kwargs))

def ishear_right_down(images, **kwargs):
    images = tf.image.flip_up_down(images)
    return tf.image.flip_up_down(shear_left_down(images, **kwargs))

def ishear_rot90_down(images, **kwargs):
    images = tf.image.flip_up_down(images)
    return tf.image.flip_up_down(shear_left_down(images, **kwargs))


#["TILT", "TILT_LEFT_RIGHT", "TILT_TOP_BOTTOM", "CORNER"]
def tilt_left_random(images, **kwargs):
    images = tf.pad(images, [[0, 0], [kwargs['height']//10, kwargs['height']//10],
                             [kwargs['width']//10, kwargs['width']//10], [0, 0]], 'CONSTANT')
    images = tf.image.resize(tfa.image.transform(images,
                                                 kwargs['skew_matrix'],
                                                 interpolation="NEAREST"),
                             (kwargs['height'], kwargs['width']))

    return  fix_bourders(images, **kwargs)


def tilt_up_random(images, **kwargs):
    images = tf.pad(images, [[0, 0], [kwargs['height']//10, kwargs['height']//10],
                             [kwargs['width']//10, kwargs['width']//10], [0, 0]], 'CONSTANT')

    images = rot90(images)
    images = rot90(rot90(rot90(
        tf.image.resize(tfa.image.transform(images, kwargs['skew_matrix'],
                                            interpolation="NEAREST"),
                        (kwargs['height'], kwargs['width'])))
    ))

    return fix_bourders(images, **kwargs)




def enhance_shape(images, prc=10):
    def _py_enhance_shape(img):
        images = []
        for i in range(len(img)):
            images += [cv2.detailEnhance(cv2.cvtColor(img[i].numpy().astype(np.uint8), cv2.IMREAD_COLOR),
                                         10, prc)]
        return np.array(images)

    return tf.py_function(_py_enhance_shape, [images], tf.float32)


def expand_background(images):
    def _py_extract_background(imgs):
        images = []
        k = -1
        dx = 0
        dy = 0
        for i in range(len(imgs)):
            img = cv2.cvtColor(imgs[i].numpy().astype(np.uint8), cv2.IMREAD_COLOR)
            height, width = img.shape[:2]
            padding = 50
            img = cv2.copyMakeBorder(img, padding, padding, padding, padding, cv2.BORDER_REPLICATE)
            img = cv2.resize(img, (height, width))

            k = k * 0.00001
            dx = dx * width
            dy = dy * height
            x, y = np.mgrid[0:width:1, 0:height:1]
            x = x.astype(np.float32) - width / 2 - dx
            y = y.astype(np.float32) - height / 2 - dy
            theta = np.arctan2(y, x)
            d = (x * x + y * y) ** 0.5
            r = d * (1 + k * d * d)
            map_x = r * np.cos(theta) + width / 2 + dx
            map_y = r * np.sin(theta) + height / 2 + dy

            images += [cv2.remap(img, map_y, map_x, interpolation=cv2.INTER_LINEAR,
                                                   borderMode=cv2.BORDER_REPLICATE)]

        return np.array(images)

    return tf.py_function(_py_extract_background, [images], tf.float32)


def fix_bourders(images, **kwargs):
    mask = tf.pad(images, [[0, 0], [kwargs['height'] // 50, kwargs['height'] // 50],
                         [kwargs['width'] // 50, kwargs['width'] // 50], [0, 0]], 'CONSTANT')
    mask = tf.image.resize(mask, (kwargs['height'], kwargs['width']))
    mask = tf.image.rgb_to_grayscale(mask)
    mask = tf.where(mask == 0, 255+tf.zeros_like(mask), mask)
    mask = tf.image.grayscale_to_rgb(tf.where(mask != 255, tf.zeros_like(mask), mask))
    mask = tf.where(mask == 255, tf.ones_like(mask), mask)
    mask = images * mask
    mask = tf.where(mask != 0, tf.ones_like(mask), mask)
    mask = tf.ones_like(mask) - mask
    mask = tf.where(mask != 1, tf.zeros_like(mask), mask)
    images = images * mask
    return inpaint(images, tf.where(images==0, tf.zeros_like(images), tf.ones_like(images)))


@tf.function
def rot90(images):
    return tf.transpose(tf.reverse(images, [2]), [0, 2, 1, 3])

def transformImg(imgIn,forward_transform):
    t = tfa.image.transform_ops.matrices_to_flat_transforms(tf.linalg.inv(forward_transform))
    return tfa.image.transform(imgIn, t, interpolation="BILINEAR")

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
    return tf.cast(perspective_skew_coefficients_matrix, tf.float32)


