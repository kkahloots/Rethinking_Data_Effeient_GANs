import cv2
import numpy as np
import tensorflow as tf

flags = sorted(list(range(0, 8)) + list(range(32, 36)) + list(range(60, 62)) + list(range(72, 74)) + \
               list(range(10, 12)))

def color_space_transform(images, **kwargs):
    def _py_color_space(img):
        images = []
        for i in range(len(img)):
            f = kwargs['flag']
            images += [simplest_cb(cv2.cvtColor(cv2.cvtColor(cv2.cvtColor(
                img[i].numpy().astype(np.uint8), cv2.IMREAD_COLOR), f), cv2.IMREAD_COLOR))]
        return np.array(images)

    return tf.py_function(_py_color_space, [images], tf.float32)


def simplest_cb(img, percent=2):
    out_channels = []
    cumstops = (
        img.shape[0] * img.shape[1] * percent / 200.0,
        img.shape[0] * img.shape[1] * (1 - percent / 200.0)
    )
    for channel in cv2.split(img):
        cumhist = np.cumsum(cv2.calcHist([channel], [0], None, [256], (0,256)))
        low_cut, high_cut = np.searchsorted(cumhist, cumstops)
        lut = np.concatenate((
            np.zeros(low_cut),
            np.around(np.linspace(0, 255, high_cut - low_cut + 1)),
            255 * np.ones(255 - high_cut)
        ))
        out_channels.append(cv2.LUT(channel, lut.astype('uint8')))
    return cv2.merge(out_channels)
