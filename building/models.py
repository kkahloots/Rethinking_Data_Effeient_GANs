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

"""Implementation of WGANGP model.

Details available at https://arxiv.org/abs/1704.00028.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import tempfile
from functools import partial
from livelossplot.plot_losses import PlotLosses

import tensorflow as tf
from tensorflow import random
from tensorflow.python.keras import layers
from tensorflow.python.keras import metrics
from tensorflow.python.keras import models

import building.ops as ops
from utils.utils import img_merge
from utils.utils import pbar, monitor_generator
from utils.utils import save_image_grid



class WGAN_GP:
    def __init__(self,
                 model_name,
                 image_size,
                 save_path=None,
                 batch_size=36,
                 z_dim=256,
                 n_critic=5,
                 g_penalty=10,
                 g_lr=0.0001,
                 d_lr=0.0001):
        self.model_name = model_name
        self.save_path = save_path
        self.z_dim = z_dim
        self.batch_size = batch_size
        self.image_size = image_size
        self.n_critic = n_critic
        self.grad_penalty_weight = g_penalty
        self.g_opt = ops.AdamOptWrapper(learning_rate=g_lr)
        self.d_opt = ops.AdamOptWrapper(learning_rate=d_lr)

        self.G = self.build_generator()
        self.D = self.build_discriminator()

        try:
            self.G.load_weights(filepath=f'{self.save_path}/{self.model_name}_generator')
            print('restore generator successfully ... ')

            self.D.load_weights(filepath=f'{self.save_path}/{self.model_name}_discriminator')
            print('restore discriminator successfully ... ')
        except:
            print('unable to restore ... ')

        self.G.summary()
        self.D.summary()

    def train(self, dataset, epochs=50, n_itr=100, min_delta=1e-9):
        z = tf.constant(random.normal((self.batch_size, 1, 1, self.z_dim)))
        g_train_loss = metrics.Mean()
        d_train_loss = metrics.Mean()
        liveplot = PlotLosses()
        stop_training, best, wait = False, 1e-9, 0

        for epoch in range(epochs):
            bar = pbar(n_itr, epoch, epochs)
            for itr_c, batch in zip(range(n_itr), dataset):
                if itr_c >= n_itr:
                    print('yes')
                    bar.close()
                    break

                for _ in range(self.n_critic):
                    self.train_d(batch['images'])
                    d_loss = self.train_d(batch['images'])
                    d_train_loss(d_loss)

                g_loss = self.train_g()
                g_train_loss(g_loss)
                self.train_g()

                bar.postfix['g_loss'] = f'{g_train_loss.result():6.3f}'
                bar.postfix['d_loss'] = f'{d_train_loss.result():6.3f}'
                bar.update(itr_c)

            bar.close()
            current = g_train_loss.result()
            losses = {'g_loss': g_train_loss.result(), 'd_loss': d_train_loss.result()}
            liveplot.update(losses, epoch)
            liveplot.send()

            g_train_loss.reset_states()
            d_train_loss.reset_states()
            del bar

            self.G.save_weights(filepath=f'{self.save_path}/{self.model_name}_generator')
            self.D.save_weights(filepath=f'{self.save_path}/{self.model_name}_discriminator')

            samples = self.generate_samples(z)
            image_grid = img_merge(samples, n_rows=6).squeeze()
            img_path = f'./images/{self.model_name}'
            os.makedirs(img_path, exist_ok=True)
            save_image_grid(image_grid, epoch + 1, self.model_name, output_dir=img_path)

            stop_training, best, wait = monitor_generator(epoch, wait, min_delta, current, best, self.G)

    @tf.function
    def train_g(self):
        z = random.normal((self.batch_size, 1, 1, self.z_dim))
        with tf.GradientTape() as t:
            x_fake = self.G(z, training=True)
            fake_logits = self.D(x_fake, training=True)
            loss = ops.g_loss_fn(fake_logits)
        grad = t.gradient(loss, self.G.trainable_variables)
        self.g_opt.apply_gradients(zip(grad, self.G.trainable_variables))
        return loss

    @tf.function
    def train_d(self, x_real):
        z = random.normal((self.batch_size, 1, 1, self.z_dim))
        with tf.GradientTape() as t:
            x_fake = self.G(z, training=True)
            fake_logits = self.D(x_fake, training=True)
            real_logits = self.D(x_real, training=True)
            cost = ops.d_loss_fn(fake_logits, real_logits)
            gp = self.gradient_penalty(partial(self.D, training=True), x_real, x_fake)
            cost += self.grad_penalty_weight * gp
        grad = t.gradient(cost, self.D.trainable_variables)
        self.d_opt.apply_gradients(zip(grad, self.D.trainable_variables))
        return cost

    def gradient_penalty(self, f, real, fake):
        alpha = random.uniform([self.batch_size, 1, 1, 1], 0., 1.)
        diff = fake - real
        inter = real + (alpha * diff)
        with tf.GradientTape() as t:
            t.watch(inter)
            pred = f(inter)
        grad = t.gradient(pred, [inter])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=[1, 2, 3]))
        gp = tf.reduce_mean((slopes - 1.)**2)
        return gp

    @tf.function
    def generate_samples(self, z):
        """Generates sample images using random values from a Gaussian distribution."""
        return self.G(z, training=False)

    def build_generator(self):
        dim = self.image_size[0]
        mult = dim // 8

        x = inputs = layers.Input((1, 1, self.z_dim))
        x = ops.UpConv2D(dim * mult, 5, 1, 'valid')(x)
        x = ops.BatchNorm()(x)
        x = layers.ReLU()(x)

        x = ops.UpConv2D(dim * mult, 4, 5, 'valid')(x)
        x = ops.BatchNorm()(x)
        x = layers.ReLU()(x)

        x = ops.UpConv2D(dim * (mult // 2), kernel_size=4, strides=2)(x)
        x = ops.BatchNorm()(x)
        x = layers.ReLU()(x)
        mult //= 2

        x = ops.UpConv2D(3)(x)
        x = layers.Activation('tanh')(x)
        return models.Model(inputs, x, name='Generator')


    def build_discriminator(self):
        dim = self.image_size[0]
        mult = 1
        i = dim // 2

        x = inputs = layers.Input((dim, dim, 3))
        x = ops.Conv2D(dim)(x)
        x = ops.LeakyRelu()(x)

        while i > 4:
            x = ops.Conv2D(dim * (2 * mult))(x)
            x = ops.LayerNorm(axis=[1, 2, 3])(x)
            x = ops.LeakyRelu()(x)

            i //= 2
            mult *= 2

        x = ops.Conv2D(1, 4, 1, 'valid')(x)
        return models.Model(inputs, x, name='Discriminator')
