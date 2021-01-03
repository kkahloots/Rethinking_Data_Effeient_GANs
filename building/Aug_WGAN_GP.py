
import os
from functools import partial
from livelossplot.plot_losses import PlotLosses

import tensorflow as tf
import numpy as np
import pickle
from tensorflow import random
from tensorflow.python.keras import layers
from tensorflow.python.keras import metrics
from tensorflow.python.keras import models

import building.ops as ops
from utils.utils import img_merge
from utils.utils import pbar, vbar
from utils.utils import save_image_grid
from augmentation.augmentor import Augmentor

class Augmented_WGAN_GP:
    def __init__(self,
                 model_name,
                 image_shape,
                 image_scale=255.0,
                 td_prob=0.25,
                 aug_level=4,
                 save_path=None,
                 batch_size=36,
                 z_dim=256,
                 n_critic=5,
                 g_penalty=10,
                 g_lr=0.0001,
                 d_lr=0.0001):

        self.model_name = model_name
        self.Augmentor = Augmentor()
        self.td_prob =td_prob
        self.save_path = save_path
        self.z_dim = z_dim
        self.batch_size = batch_size
        self.aug_level = aug_level
        self.image_shape = image_shape
        self.image_scale = image_scale
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

    def train(self, dataset, val_dataset=None, epochs=int(6e4), n_itr=100, plot_live=True):
        try:
            z = tf.constant(np.load(f'{self.save_path}/{self.model_name}_z.npy'))
        except FileNotFoundError:
            z = tf.constant(random.normal((self.batch_size, 1, 1, self.z_dim)))
            os.makedirs(self.save_path, exist_ok=True)
            np.save(f'{self.save_path}/{self.model_name}_z', z.numpy())

        if plot_live:
            liveplot = PlotLosses()
        try:
            losses_list = pickle.load(open(f'{self.save_path}/{self.model_name}_losses_list.pkl', 'rb'))
        except:
            losses_list = []

        if plot_live:
            for i, losses in enumerate(losses_list):
                liveplot.update(losses, i)
        
        start_epoch = len(losses_list)    

        g_train_loss = metrics.Mean()
        d_train_loss = metrics.Mean()
        d_val_loss = metrics.Mean()


        for epoch in range(start_epoch, epochs):
            train_bar = pbar(n_itr, epoch, epochs)
            for itr_c, batch in zip(range(n_itr), dataset):
                if train_bar.n >= n_itr:
                    break

                for _ in range(self.n_critic):
                    d_loss = self.train_d(batch, image_scale=self.image_scale)
                    d_train_loss(d_loss)

                g_loss = self.train_g(image_scale=self.image_scale)
                g_train_loss(g_loss)

                train_bar.postfix['g_loss'] = f'{g_train_loss.result():6.3f}'
                train_bar.postfix['d_loss'] = f'{d_train_loss.result():6.3f}'
                train_bar.update(n=itr_c)

            train_bar.close()

            if val_dataset is not None:
                val_bar = vbar(n_itr//5, epoch, epochs)
                for itr_c, batch in zip(range(n_itr//5), val_dataset):
                    if val_bar.n >= n_itr//5:
                        break

                    d_val_l = self.val_d(batch , image_scale=self.image_scale)
                    d_val_loss(d_val_l)

                    val_bar.postfix['d_val_loss'] = f'{d_val_loss.result():6.3f}'
                    val_bar.update(n=itr_c)
                val_bar.close()


            losses = {'g_loss': g_train_loss.result(),
                      'd_loss': d_train_loss.result(),
                      'd_val_loss': d_val_loss.result()}
            losses_list += [losses]
            pickle.dump(losses_list, open(f'{self.save_path}/{self.model_name}_losses_list.pkl', 'wb'))
            if plot_live:
                liveplot.update(losses, epoch)
                liveplot.send()

            g_train_loss.reset_states()
            d_train_loss.reset_states()
            d_val_loss.reset_states()


            self.G.save_weights(filepath=f'{self.save_path}/{self.model_name}_generator')
            self.D.save_weights(filepath=f'{self.save_path}/{self.model_name}_discriminator')

            if epoch >= int(2e4):
                if epoch%1000 == 0:
                    self.G.save_weights(filepath=f'{self.save_path}/{self.model_name}_generator{epoch}')
                    self.D.save_weights(filepath=f'{self.save_path}/{self.model_name}_discriminator{epoch}')

            if epoch%250 ==1:
                samples = self.generate_samples(z, image_scale=self.image_scale)
                img_path = f'{self.save_path}/images/{self.model_name}'
                os.makedirs(img_path, exist_ok=True)
                image_grid = img_merge(samples.numpy(), n_rows=6).squeeze()
                save_image_grid(image_grid, epoch, self.model_name, output_dir=img_path)


    @tf.function
    def train_g(self, image_scale=255.0):
        z = random.normal((self.batch_size, 1, 1, self.z_dim))
        with tf.GradientTape() as t:
            x_fake = self.G(z, training=True) * image_scale
            fake_logits = self.D(self.Augmentor.augment(images=x_fake, td_prob=self.td_prob, scale=image_scale,\
                                                batch_shape=[self.batch_size, *self.image_shape]), training=True)
            loss = ops.g_loss_fn(fake_logits)
        grad = t.gradient(loss, self.G.trainable_variables)
        self.g_opt.apply_gradients(zip(grad, self.G.trainable_variables))
        return loss

    @tf.function
    def train_d(self, x_real, image_scale=255.0):
        z = random.normal((self.batch_size, 1, 1, self.z_dim))
        with tf.GradientTape() as t:
            x_fake = self.G(z, training=True) * image_scale
            fake_logits = self.D(self.Augmentor.augment(images=x_fake, td_prob=self.td_prob, scale=image_scale, \
                                                batch_shape=[self.batch_size, *self.image_shape]), training=True)
            real_logits = self.D(self.Augmentor.augment(images=x_real, td_prob=self.td_prob, scale=image_scale, \
                                                batch_shape=[self.batch_size, *self.image_shape]), training=True)
            cost = ops.d_loss_fn(fake_logits, real_logits)
            gp = self.gradient_penalty(partial(self.D, training=True), x_real/image_scale, x_fake/image_scale)
            cost += self.grad_penalty_weight * gp
        grad = t.gradient(cost, self.D.trainable_variables)
        self.d_opt.apply_gradients(zip(grad, self.D.trainable_variables))
        return cost


    @tf.function
    def val_d(self, x_real, image_scale=255.0):
        z = random.normal((self.batch_size, 1, 1, self.z_dim))
        x_fake = self.G(z, training=False) * image_scale
        fake_logits = self.D(x_fake/image_scale, training=False)
        real_logits = self.D(x_real/image_scale, training=False)
        cost = ops.d_loss_fn(fake_logits, real_logits)
        gp = self.gradient_penalty(partial(self.D, training=False), x_real/image_scale, x_fake/image_scale)
        cost += self.grad_penalty_weight * gp
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
    def generate_samples(self, z, image_scale=255.0):
        """Generates sample images using random values from a Gaussian distribution."""
        return self.G(z, training=False) * image_scale


    def build_generator(self):
        dim = self.image_shape[0]
        mult = dim // 8

        x = inputs = layers.Input((1, 1, self.z_dim))
        x = ops.UpConv2D(dim//2 * mult, 4, 1, 'valid')(x)
        x = ops.BatchNorm()(x)
        x = layers.ReLU()(x)

        while mult > 1:
            x = ops.UpConv2D(dim//2 * (mult // 2))(x)
            x = ops.BatchNorm()(x)
            x = layers.ReLU()(x)

            mult //= 2

        x = ops.UpConv2D(3)(x)
        x = layers.Activation('tanh')(x)
        return models.Model(inputs, x, name='Generator')

    def build_discriminator(self):
        dim = self.image_shape[0]
        mult = 1
        i = dim // 2

        x = inputs = layers.Input((dim, dim, 3))
        x = ops.Conv2D(dim//2)(x)
        x = ops.LeakyRelu()(x)

        while i > 4:
            x = ops.Conv2D(dim//2 * (2 * mult))(x)
            x = ops.LayerNorm(axis=[1, 2, 3])(x)
            x = ops.LeakyRelu()(x)

            i //= 2
            mult *= 2

        x = ops.Conv2D(1, 4, 1, 'valid')(x)
        return models.Model(inputs, x, name='Discriminator')
