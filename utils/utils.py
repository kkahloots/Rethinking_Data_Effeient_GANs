# MIT License
#
# Copyright (c) 2019 Drew Szurko
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import shutil

import numpy as np
import tensorflow as tf
from tqdm.autonotebook import tqdm


def img_merge(images, n_rows=None, n_cols=None, padding=0, pad_value=0):
    #images = np.array(images)
    n = images.shape[0]

    if n_rows:
        n_rows = max(min(n_rows, n), 1)
        n_cols = int(n - 0.5) // n_rows + 1
    elif n_cols:
        n_cols = max(min(n_cols, n), 1)
        n_rows = int(n - 0.5) // n_cols + 1
    else:
        n_rows = int(n**0.5)
        n_cols = int(n - 0.5) // n_rows + 1

    h, w = images.shape[1], images.shape[2]
    shape = (h * n_rows + padding * (n_rows - 1), w * n_cols + padding * (n_cols - 1))
    if images.ndim == 4:
        shape += (images.shape[3], )
    img = np.full(shape, pad_value, dtype=images.dtype)

    for idx, image in enumerate(images):
        i = idx % n_cols
        j = idx // n_cols
        img[j * (h + padding):j * (h + padding) + h, i * (w + padding):i *
            (w + padding) + w, ...] = image

    return img.astype(np.uint8)


def save_image_grid(img_grid, epoch, model_name, output_dir):
    """Saves image grid to user output dir."""
    file_name = model_name + f'_{epoch}.jpg'
    output_dir = os.path.join(output_dir, file_name)
    tf.io.write_file(output_dir, tf.image.encode_jpeg(tf.cast(img_grid, tf.uint8)))


def get_terminal_width():
    width = shutil.get_terminal_size(fallback=(200, 24))[0]
    if width == 0:
        width = 120
    return width


class mybar(tqdm):
    @property
    def n(self):
        return self.__n

    @n.setter
    def n(self, value):
        # if hasattr(self, 'n'):
        #     new_value = value%self.total
        # else:
        #     new_value = value

        #if value > self.total:
        #    self.reset(total=self.total)


        self.__n = value

    # def update(self, *args, **kwargs):
    #     if not hasattr(self, 'ototal'):
    #          self.ototal = self.total
    #     super(tqdm, self).update(*args, **kwargs)
    #     return  kwargs['n'] > self.ototal


def vbar(total_images, epoch, epochs):
    bar = mybar(total=total_images,
               ncols=int(get_terminal_width() * .9),
               desc=tqdm.write(f'Epoch {epoch + 1}/{epochs}'),
               postfix={
                   'd_val_loss': f'{0:6.3f}',
                   1: 1
               },
               bar_format='{n_fmt}/{total_fmt} |{bar}| {rate_fmt}  '
               'ETA: {remaining}  Elapsed Time: {elapsed}  '
               'D Loss: {postfix[d_val_loss]}',
               unit=' images',
               miniters=10)
    return bar

def pbar(total_images, epoch, epochs):
    bar = mybar(total=total_images,
               ncols=int(get_terminal_width() * .9),
               desc=tqdm.write(f'Epoch {epoch + 1}/{epochs}'),
               postfix={
                   'g_loss': f'{0:6.3f}',
                   'd_loss': f'{0:6.3f}',
                   1: 1
               },
               bar_format='{n_fmt}/{total_fmt} |{bar}| {rate_fmt}  '
               'ETA: {remaining}  Elapsed Time: {elapsed}  \n'
               'G Loss: {postfix[g_loss]}  \n D Loss: {postfix[d_loss]}',
               unit=' images',
               miniters=10)
    return bar


def monitor_generator(epoch,
                      wait,
                      min_delta,
                      current,
                      best,
                      model,
                      mode='auto',
                      patience=9,
                      verbose=1,
                      restore_best_weights=True):
    stop_training = False

    if mode not in ['auto', 'min', 'max']:
      mode = 'auto'

    if mode == 'min':
      monitor_op = np.less
    elif mode == 'max':
      monitor_op = np.greater
    else:
        monitor_op = np.less

    if monitor_op == np.greater:
      min_delta *= 1
    else:
      min_delta *= -1

    best_weights = model.get_weights()

    if monitor_op(current - min_delta, best):
      best = current
      wait = 0

    else:
      wait += 1
      if wait >= patience:
        stopped_epoch = epoch
        stop_training = True
        if restore_best_weights:
          if verbose > 0:
            print('Restoring model weights from the end of the best epoch.')
          model.set_weights(best_weights)

    return stop_training, best, wait
