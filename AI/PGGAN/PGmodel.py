from tensorflow.keras import Model
from abc import ABC
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Dense, \
    Flatten


def fn(level):
    if level > 5:
        return 2 ** (14 - level)
    else:
        return 512


# 上采样
def upsampling2d(ginput):
    h = ginput.shape[1]
    w = ginput.shape[2]
    return tf.image.resize(ginput, size=[2 * h, 2 * w], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)


# 下采样
def downsampling2d(dinput):
    return tf.nn.avg_pool2d(dinput, ksize=2, strides=2, padding='VALID')


# 像素归一化
def pn(pinput):
    return pinput / tf.sqrt(tf.reduce_mean(tf.square(pinput), axis=3, keepdims=True) + 1e-8)


# 添加多样性特征
def minibatchcontact(inputs):
    s = inputs.shape
    vals = tf.sqrt(tf.reduce_mean((inputs - tf.reduce_mean(inputs, axis=0, keepdims=True)) ** 2, axis=0, keepdims=True)
                   + 1e-8)
    vals = tf.reduce_mean(vals, keepdims=True)
    vals = tf.tile(vals, multiples=(s[0], s[1], s[2], 1))
    return tf.concat([inputs, vals], axis=3)


# 生成器模型
class GconvBlock(Model, ABC):

    def __init__(self, filters):
        super(GconvBlock, self).__init__()
        if filters < 512:
            self.c1 = Conv2D(filters=filters, kernel_size=3, activation=tf.nn.leaky_relu, padding='SAME',
                             kernel_initializer=tf.keras.initializers.HeNormal)
            self.c2 = Conv2D(filters=filters, kernel_size=3, activation=tf.nn.leaky_relu, padding='SAME',
                             kernel_initializer=tf.keras.initializers.HeNormal)
        else:
            self.c1 = Conv2D(filters=filters, kernel_size=3, activation=tf.nn.leaky_relu, padding='SAME',
                             kernel_initializer=tf.keras.initializers.HeNormal)
            self.c2 = Conv2D(filters=filters, kernel_size=3, activation=tf.nn.leaky_relu, padding='SAME',
                             kernel_initializer=tf.keras.initializers.HeNormal)

    def call(self, inputs, training=None, mask=None):
        x = upsampling2d(inputs)
        x = self.c1(x)
        x = pn(x)
        x = self.c2(x)
        x = pn(x)
        return x


class Generator(Model, ABC):

    def __init__(self, levels):
        super(Generator, self).__init__()
        self.f1 = Dense(512 * 4 * 4, activation=tf.nn.leaky_relu,
                        kernel_initializer=tf.keras.initializers.HeNormal)
        self.c1 = Conv2D(filters=512, kernel_size=3, activation=tf.nn.leaky_relu, padding='SAME',
                         kernel_initializer=tf.keras.initializers.HeNormal)
        self.rgb = []
        self.cblocks = []
        for level in levels:
            self.rgb.append(Conv2D(filters=3, kernel_size=1, padding='SAME',
                                   kernel_initializer=tf.keras.initializers.HeNormal))
            if level > 2:
                self.cblocks.append(GconvBlock(fn(level)))

    def generate(self, inputs, level, transit=False, trans_alpha=0):
        x = self.f1(inputs)
        x = tf.reshape(x, [-1, 4, 4, 512])
        x = pn(x)
        x = self.c1(x)
        x = pn(x)
        rgb0 = x
        if level == 2:
            out = self.rgb[level - 2](x)
            return out
        else:
            for i in range(level - 2):
                rgb0 = x
                x = self.cblocks[i](x)
            x = self.rgb[level - 2](x)
            if transit:
                rgb0 = upsampling2d(rgb0)
                rgb0 = self.rgb[level - 3](rgb0)
                out = trans_alpha * x + (1 - trans_alpha) * rgb0
                return out
            else:
                out = x
                return out


# 判别器模型
class DconvBlock(Model, ABC):

    def __init__(self, filters):
        super(DconvBlock, self).__init__()
        if filters < 512:
            self.c1 = Conv2D(filters=filters, kernel_size=3, activation=tf.nn.leaky_relu, padding='SAME',
                             kernel_initializer=tf.keras.initializers.HeNormal)
            self.c2 = Conv2D(filters=filters * 2, kernel_size=3, activation=tf.nn.leaky_relu, padding='SAME',
                             kernel_initializer=tf.keras.initializers.HeNormal)
        else:
            self.c1 = Conv2D(filters=filters, kernel_size=3, activation=tf.nn.leaky_relu, padding='SAME',
                             kernel_initializer=tf.keras.initializers.HeNormal)
            self.c2 = Conv2D(filters=filters, kernel_size=3, activation=tf.nn.leaky_relu, padding='SAME',
                             kernel_initializer=tf.keras.initializers.HeNormal)

    def call(self, inputs, training=None, mask=None):
        x = self.c1(inputs)
        x = pn(x)
        x = self.c2(x)
        x = pn(x)
        x = downsampling2d(x)
        return x


class Discriminator(Model, ABC):

    def __init__(self, levels):
        super(Discriminator, self).__init__()
        self.rgb = []
        self.cblocks = []
        for level in levels:
            self.rgb.append(Conv2D(filters=fn(level), kernel_size=1, padding='SAME',
                                   kernel_initializer=tf.keras.initializers.HeNormal))
            if level > 2:
                self.cblocks.append(DconvBlock(fn(level)))
        self.c1 = Conv2D(filters=512, kernel_size=3, activation=tf.nn.leaky_relu, padding='SAME',
                         kernel_initializer=tf.keras.initializers.HeNormal)
        self.c2 = Conv2D(filters=512, kernel_size=4, activation=tf.nn.leaky_relu,
                         kernel_initializer=tf.keras.initializers.HeNormal)
        self.f1 = Flatten()
        self.f2 = Dense(1, kernel_initializer=tf.keras.initializers.HeNormal)

    def discriminate(self, inputs, level, transit=False, trans_alpha=0):
        rgb0 = inputs
        x = self.rgb[level - 2](inputs)

        if level > 2:
            if transit:
                rgb0 = downsampling2d(rgb0)
                rgb0 = self.rgb[level - 3](rgb0)
                x = self.cblocks[level - 3](x)
                x = trans_alpha * x + (1 - trans_alpha) * rgb0
            else:
                x = self.cblocks[level - 3](x)
            for i in range(level - 3):
                x = self.cblocks[level - 4 - i](x)

        x = minibatchcontact(x)
        x = self.c1(x)
        x = pn(x)
        x = self.c2(x)
        x = pn(x)
        x = self.f1(x)
        out = self.f2(x)
        return out
