import cv2
import imutils
import random
import numpy as np
import tensorflow as tf


def _sample_ROI(image):
    bg = image.copy()
    bg = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
    bg = cv2.GaussianBlur(bg, (1, 1), 0)
    bg = cv2.Canny(bg, 150, 255, 0)

    cnts = cv2.findContours(bg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:25]

    ROIs = []
    for c in cnts[::-1]:
        x, y, w, h = cv2.boundingRect(c)
        approx = cv2.approxPolyDP(c, 0, True)
        ROIs.append([x, y, w, h, approx])

    ROIs_sample = random.sample(ROIs, random.choice(range(min(5, len(ROIs)), len(ROIs) + 1)))
    return ROIs_sample

def _color_bg(image, bg, ROIs):
    radius = 10 # bg.shape[0] // 2

    for x, y, w, h, approx in ROIs:
        mask = np.zeros(bg.shape[:2], np.uint8)
        cv2.drawContours(mask, [cv2.convexHull(approx)], -1, (255, 255, 255), -1, cv2.LINE_AA)
        obj = cv2.bitwise_and(image, image, mask=mask)
        ix = np.where(obj != 0)
        bg[ix] = 255

        mask = cv2.cvtColor(obj, cv2.COLOR_BGR2GRAY)
        bg = cv2.inpaint(bg, mask, radius, flags=cv2.INPAINT_TELEA)

    return bg


def _resize_place_ROIs(image, bg, ROIs, scales):
    for x, y, w, h, approx in ROIs:
        mask = np.zeros(bg.shape[:2], np.uint8)
        cv2.drawContours(mask, [cv2.convexHull(approx)], -1, (255, 255, 255), -1, cv2.LINE_AA)
        obj = cv2.bitwise_and(image, image, mask=mask)

        ix = np.where(obj == 0)
        obj[ix] = 255

        scale = (float(random.choice(scales)), float(random.choice(scales)))  # define your scale
        scaled_obj = cv2.resize(obj[y:y + h, x:x + w], None, fx=scale[0], fy=scale[1])  # scale image

        ix = np.where(scaled_obj == 255)
        scaled_obj[ix] = 0

        cv2.floodFill(scaled_obj, None, (0, 0), 0)
        kernel = np.ones((3, 3), np.uint8)
        scaled_obj = cv2.erode(scaled_obj, kernel, iterations=1)
        scaled_obj = cv2.dilate(scaled_obj, kernel, iterations=1)

        sh, sw = scaled_obj.shape[:2]  # get h, w of scaled image

        padded_scaled = np.zeros(bg.shape, dtype=np.uint8)  # using img.shape to obtain #channels
        pw, ph, _ = padded_scaled[y:y + sh, x:x + sw].shape
        padded_scaled[y:y + sh, x:x + sw] = cv2.resize(scaled_obj, (ph, pw))

        ix = np.where(padded_scaled != 0)
        bg[ix] = padded_scaled[ix]

    return bg


def _resize_place_ROIs_with_BB(image, bg, ROIs, scales):
    for x, y, w, h, approx in ROIs:
        mask = np.zeros(bg.shape[:2], np.uint8)
        cv2.drawContours(mask, [cv2.convexHull(approx)], -1, (255, 255, 255), -1, cv2.LINE_AA)
        obj = cv2.bitwise_and(image, image, mask=mask)

        ix = np.where(obj == 0)
        obj[ix] = 255

        scale = (float(random.choice(scales)), float(random.choice(scales)))  # define your scale
        scaled_obj = cv2.resize(obj[y:y + h, x:x + w], None, fx=scale[0], fy=scale[1])  # scale image

        ix = np.where(scaled_obj == 255)
        scaled_obj[ix] = 0

        sh, sw = scaled_obj.shape[:2]  # get h, w of scaled image

        padded_scaled = np.zeros(bg.shape, dtype=np.uint8)  # using img.shape to obtain #channels
        pw, ph, _ = padded_scaled[y:y + sh, x:x + sw].shape
        padded_scaled[y:y + sh, x:x + sw] = cv2.resize(scaled_obj, (ph, pw))
        ix = np.where(padded_scaled != 0)
        bg[ix] = padded_scaled[ix]

        cv2.rectangle(bg, (x, y), (x+sw, y+sh), (36, 255, 12), 1)

    return bg


def aug_bg_patches(images, scales, aug_fun):

    def _py_detect_patches(images, scales):
        images = images.numpy().astype(np.uint8)
        bgs = []
        for image in images:
            ROIs_sample = _sample_ROI(image)
            bg = _color_bg(image, image.copy(), ROIs_sample)
            bg = cv2.cvtColor(aug_fun(tf.expand_dims(bg, 0)).numpy()[0].astype(np.uint8), cv2.IMREAD_COLOR)
            bg = _resize_place_ROIs(image, bg, ROIs_sample, scales)

            bgs += [bg]
        return np.stack(bgs)

    augmented = tf.py_function(_py_detect_patches, [images, scales], tf.float32)
    return tf.reshape(augmented, tf.shape(images))


def aug_bg_patches_demo(images, scales, aug_fun):

    def _py_detect_patches(images, scales):
        images = images.numpy().astype(np.uint8)
        bgs = []
        bgsBB = []
        for image in images:
            ROIs_sample = _sample_ROI(image)
            bg = _color_bg(image, image.copy(), ROIs_sample)
            bg = cv2.cvtColor(aug_fun(tf.expand_dims(bg, 0)).numpy()[0].astype(np.uint8), cv2.IMREAD_COLOR)
            bgBB = _resize_place_ROIs_with_BB(image, bg.copy(), ROIs_sample, scales)
            bg = _resize_place_ROIs(image, bg, ROIs_sample, scales)

            bgs += [bg]
            bgsBB += [bgBB]
        return np.stack([bgs, bgsBB])

    augmented = tf.py_function(_py_detect_patches, [images, scales], tf.float32)
    return tf.reshape(augmented[0], tf.shape(images)), tf.reshape(augmented[1], tf.shape(images))
