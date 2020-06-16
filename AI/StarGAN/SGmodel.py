from abc import ABC

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Activation, Conv2DTranspose


class Resblock(Model, ABC):

    def __init__(self):
        super(Resblock, self).__init__()
        self.c1 = Conv2D(filters=128, kernel_size=3, padding='SAME', use_bias=False)
        self.b1 = tfa.layers.InstanceNormalization()
        self.a1 = Activation('relu')
        self.c2 = Conv2D(filters=128, kernel_size=3, padding='SAME', use_bias=False)
        self.b2 = tfa.layers.InstanceNormalization()

    def call(self, inputs, training=None, mask=None):
        x = self.c1(inputs)
        x = self.b1(x)
        x = self.a1(x)
        x = self.c2(x)
        x = self.b2(x)
        return x + inputs


# 生成器
class Generator(Model, ABC):

    def __init__(self):
        super(Generator, self).__init__()

        self.c1 = Conv2D(filters=32, kernel_size=7, padding='SAME', use_bias=False)
        self.b1 = tfa.layers.InstanceNormalization()
        self.a1 = Activation('relu')

        self.c2 = Conv2D(filters=64, kernel_size=4, strides=2, padding='SAME', use_bias=False)
        self.b2 = tfa.layers.InstanceNormalization()
        self.a2 = Activation('relu')
        self.c3 = Conv2D(filters=128, kernel_size=4, strides=2, padding='SAME', use_bias=False)
        self.b3 = tfa.layers.InstanceNormalization()
        self.a3 = Activation('relu')

        self.res = tf.keras.models.Sequential()
        for i in range(6):
            self.res.add(Resblock())

        self.c4 = Conv2DTranspose(filters=64, kernel_size=4, strides=2, padding='SAME', use_bias=False)
        self.b4 = tfa.layers.InstanceNormalization()
        self.a4 = Activation('relu')
        self.c5 = Conv2DTranspose(filters=32, kernel_size=4, strides=2, padding='SAME', use_bias=False)
        self.b5 = tfa.layers.InstanceNormalization()
        self.a5 = Activation('relu')

        self.c6 = Conv2D(filters=3, kernel_size=7, padding='SAME', use_bias=False, activation='tanh')

    def generate(self, x, c):

        c = tf.cast(tf.reshape(c, shape=[-1, 1, 1, c.shape[-1]]), tf.float32)
        c = tf.tile(c, [1, x.shape[1], x.shape[2], 1])
        x = tf.concat([x, c], axis=-1)

        x = self.c1(x)
        x = self.b1(x)
        x = self.a1(x)

        x = self.c2(x)
        x = self.b2(x)
        x = self.a2(x)
        x = self.c3(x)
        x = self.b3(x)
        x = self.a3(x)

        x = self.res(x)

        x = self.c4(x)
        x = self.b4(x)
        x = self.a4(x)
        x = self.c5(x)
        x = self.b5(x)
        x = self.a5(x)

        x = self.c6(x)
        return x


# 生成器
class Discriminator(Model, ABC):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.c1 = Conv2D(filters=32, kernel_size=4, strides=2, padding='SAME')
        self.c2 = Conv2D(filters=64, kernel_size=4, strides=2, padding='SAME')
        self.c3 = Conv2D(filters=128, kernel_size=4, strides=2, padding='SAME')
        self.c4 = Conv2D(filters=256, kernel_size=4, strides=2, padding='SAME')
        self.c5 = Conv2D(filters=512, kernel_size=4, strides=2, padding='SAME')

        self.c6 = Conv2D(filters=1, kernel_size=7, padding='SAME', use_bias=False)
        self.c7 = Conv2D(filters=5, kernel_size=2, use_bias=False)

    def discriminate(self, x):
        x = self.c1(x)
        x = tf.nn.leaky_relu(x, alpha=0.01)
        x = self.c2(x)
        x = tf.nn.leaky_relu(x, alpha=0.01)
        x = self.c3(x)
        x = tf.nn.leaky_relu(x, alpha=0.01)
        x = self.c4(x)
        x = tf.nn.leaky_relu(x, alpha=0.01)
        x = self.c5(x)
        x = tf.nn.leaky_relu(x, alpha=0.01)

        logits = self.c6(x)
        label = self.c7(x)
        label = tf.reshape(label, shape=[-1, 5])
        return logits, label
