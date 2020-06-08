import os
import time

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Dense,\
    Conv2DTranspose, Dropout, MaxPool2D, GlobalAveragePooling2D


def celoss_ones(logits):
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                                   labels=tf.ones_like(logits))
    return tf.reduce_mean(loss)


def celoss_zeros(logits):
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                                   labels=tf.zeros_like(logits))
    return tf.reduce_mean(loss)


def d_loss_fn(generator, discriminator, batch_z, batch_x):
    fake_image = generator(batch_z)
    d_fake_logits = discriminator(fake_image)
    d_real_logits = discriminator(batch_x)

    d_loss_real = celoss_ones(d_real_logits)
    d_loss_fake = celoss_zeros(d_fake_logits)

    loss = d_loss_fake + d_loss_real

    return loss


def g_loss_fn(generator, discriminator, batch_z):
    fake_image = generator(batch_z)
    d_fake_logits = discriminator(fake_image)
    loss = celoss_ones(d_fake_logits)

    return loss


class Generator(Model):
    def __init__(self):
        super(Generator, self).__init__()

        self.f1 = Dense(512, activation=tf.nn.leaky_relu)
        self.f2 = Dense(2 * 2 * 512, activation=tf.nn.leaky_relu)

        self.c1 = Conv2DTranspose(512, 3, 2, padding='SAME')
        self.b1 = BatchNormalization()
        self.a1 = Activation(tf.nn.leaky_relu)
        self.c2 = Conv2DTranspose(512, 3, 1, padding='SAME')
        self.b2 = BatchNormalization()
        self.a2 = Activation(tf.nn.leaky_relu)
        self.d1 = Dropout(0.2)

        self.c3 = Conv2DTranspose(256, 3, 2, padding='SAME')
        self.b3 = BatchNormalization()
        self.a3 = Activation(tf.nn.leaky_relu)
        self.c4 = Conv2DTranspose(256, 3, 1, padding='SAME')
        self.b4 = BatchNormalization()
        self.a4 = Activation(tf.nn.leaky_relu)
        self.d2 = Dropout(0.2)

        self.c5 = Conv2DTranspose(128, 3, 2, padding='SAME')
        self.b5 = BatchNormalization()
        self.a5 = Activation(tf.nn.leaky_relu)
        self.d3 = Dropout(0.2)

        self.c6 = Conv2DTranspose(64, 3, 2, padding='SAME')
        self.b6 = BatchNormalization()
        self.a6 = Activation(tf.nn.leaky_relu)
        self.d4 = Dropout(0.2)

        self.c7 = Conv2DTranspose(32, 3, 1, padding='SAME')
        self.b7 = BatchNormalization()
        self.a7 = Activation(tf.nn.leaky_relu)

        self.c8 = Conv2DTranspose(3, 3, 1, padding='SAME')
        self.a8 = Activation('tanh')

    def call(self, inputs, training=None, mask=None):
        z = self.f1(inputs)
        z = self.f2(z)
        z = tf.reshape(z, [-1, 2, 2, 512])

        z = self.c1(z)
        z = self.b1(z)
        z = self.a1(z)
        z = self.c2(z)
        z = self.b2(z)
        z = self.a2(z)
        z = self.d1(z)

        z = self.c3(z)
        z = self.b3(z)
        z = self.a3(z)
        z = self.c4(z)
        z = self.b4(z)
        z = self.a4(z)
        z = self.d2(z)

        z = self.c5(z)
        z = self.b5(z)
        z = self.a5(z)
        z = self.d3(z)

        z = self.c6(z)
        z = self.b6(z)
        z = self.a6(z)
        z = self.d4(z)

        z = self.c7(z)
        z = self.b7(z)
        z = self.a7(z)

        z = self.c8(z)
        x = self.a8(z)
        return x


class Discriminator(Model):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.c1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')
        self.b1 = BatchNormalization()
        self.a1 = Activation(tf.nn.leaky_relu)
        self.p1 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
        self.d1 = Dropout(0.2)

        self.c2 = Conv2D(filters=128, kernel_size=(3, 3), padding='same')
        self.b2 = BatchNormalization()
        self.a2 = Activation(tf.nn.leaky_relu)
        self.p2 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
        self.d2 = Dropout(0.2)

        self.c3 = Conv2D(filters=256, kernel_size=(3, 3), padding='same')
        self.b3 = BatchNormalization()
        self.a3 = Activation(tf.nn.leaky_relu)
        self.p3 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
        self.d3 = Dropout(0.2)

        self.c4 = Conv2D(filters=512, kernel_size=(3, 3), padding='same')
        self.b4 = BatchNormalization()
        self.a4 = Activation(tf.nn.leaky_relu)
        self.c5 = Conv2D(filters=512, kernel_size=(3, 3), padding='same')
        self.b5 = BatchNormalization()
        self.a5 = Activation(tf.nn.leaky_relu)
        self.p4 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
        self.d4 = Dropout(0.2)

        self.p5 = GlobalAveragePooling2D()
        self.f1 = Dense(256, activation=tf.nn.leaky_relu)
        self.f2 = Dense(1)

    def call(self, inputs, training=None, mask=None):
        x = self.c1(inputs)
        x = self.b1(x)
        x = self.a1(x)
        x = self.p1(x)
        x = self.d1(x)

        x = self.c2(x)
        x = self.b2(x)
        x = self.a2(x)
        x = self.p2(x)
        x = self.d2(x)

        x = self.c3(x)
        x = self.b3(x)
        x = self.a3(x)
        x = self.p3(x)
        x = self.d3(x)

        x = self.c4(x)
        x = self.b4(x)
        x = self.a4(x)
        x = self.c5(x)
        x = self.b5(x)
        x = self.a5(x)
        x = self.p4(x)
        x = self.d4(x)

        x = self.p5(x)
        x = self.f1(x)
        logits = self.f2(x)
        return logits


def main():
    cifar10 = tf.keras.datasets.cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = x_train * 2 - 1
    x_train = x_train.reshape(x_train.shape[0], 32, 32, 3)
    x_train = x_train.astype('float32')

    z_dim = 300
    epochs = 50
    batch_size = 128
    learning_rate = 0.0002

    train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)

    generator = Generator()
    generator.build(input_shape=(None, z_dim))
    discriminator = Discriminator()
    discriminator.build(input_shape=(None, 32, 32, 3))

    if os.path.exists('./GAN3/generatorweights/gweights.index'):
        print('-------------load the model-----------------')
        generator.load_weights('./GAN3/generatorweights/gweights')
        discriminator.load_weights('./GAN3/discriminatorweights/dweights')

    g_optimizer = tf.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)
    d_optimizer = tf.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)

    for epoch in range(epochs):
        start1 = time.perf_counter()
        for step, (x_train, y_train) in enumerate(train_db):
            batch_z = tf.random.uniform([batch_size, z_dim], minval=-1., maxval=1.)

            with tf.GradientTape() as tape:
                d_loss = d_loss_fn(generator, discriminator, batch_z, x_train)
            grads = tape.gradient(d_loss, discriminator.trainable_variables)
            d_optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))

            with tf.GradientTape() as tape:
                g_loss = g_loss_fn(generator, discriminator, batch_z)
            grads = tape.gradient(g_loss, generator.trainable_variables)
            g_optimizer.apply_gradients(zip(grads, generator.trainable_variables))

        end1 = time.perf_counter()
        print(epoch, 'd-loss:', float(d_loss), 'g-loss:', float(g_loss), end1 - start1, 's')

    generator.save_weights('./GAN3/generatorweights/gweights')
    discriminator.save_weights('./GAN3/discriminatorweights/dweights')

    f, a = plt.subplots(4, 10, figsize=(10, 4))
    for i in range(10):
        # 随机生成噪声,噪声也是均匀分布中随机取值,用训练好的G网络生成图片
        z = tf.random.uniform(shape=[4, z_dim], maxval=1, minval=-1)
        g_sample = generator(z)
        g_sample = (g_sample + 1) / 2
        g_sample = g_sample * 255.0
        g_sample = np.reshape(g_sample, newshape=(4, 32, 32, 3)).astype(np.int)
        # 把40张图片画出来
        for j in range(4):
            a[j][i].imshow(g_sample[j])

    plt.show()


if __name__ == '__main__':
    main()
