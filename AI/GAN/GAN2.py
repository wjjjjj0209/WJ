import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Flatten, Dense, Conv2DTranspose
import time
from matplotlib import pyplot as plt


def celoss_ones(logits):
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                                   labels=tf.ones_like(logits))
    return tf.reduce_mean(loss)


def celoss_zeros(logits):
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                                   labels=tf.zeros_like(logits))
    return tf.reduce_mean(loss)


def d_loss_fn(generator, discriminator, batch_z, batch_x, is_training):
    fake_image = generator(batch_z, is_training)
    d_fake_logits = discriminator(fake_image, is_training)
    d_real_logits = discriminator(batch_x, is_training)

    d_loss_real = celoss_ones(d_real_logits)
    d_loss_fake = celoss_zeros(d_fake_logits)

    loss = d_loss_fake + d_loss_real

    return loss


def g_loss_fn(generator, discriminator, batch_z, is_training):
    fake_image = generator(batch_z, is_training)
    d_fake_logits = discriminator(fake_image, is_training)
    loss = celoss_ones(d_fake_logits)

    return loss


class Generator(Model):
    def __init__(self):
        super(Generator, self).__init__()

        self.d1 = Dense(3*3*256)
        self.a1 = Activation(tf.nn.leaky_relu)

        self.c1 = Conv2DTranspose(128, 2, 2, 'valid')
        self.b1 = BatchNormalization()
        self.a2 = Activation(tf.nn.leaky_relu)

        self.c2 = Conv2DTranspose(64, 3, 2, 'valid')
        self.b2 = BatchNormalization()
        self.a3 = Activation(tf.nn.leaky_relu)

        self.c3 = Conv2DTranspose(1, 4, 2, 'valid')
        self.a4 = Activation('tanh')

    def call(self, inputs, training=None, mask=None):
        x = self.d1(inputs)
        x = tf.reshape(x, [-1, 3, 3, 256])
        x = self.a1(x)
        x = self.c1(x)
        x = self.b1(x, training=training)
        x = self.a2(x)
        x = self.c2(x)
        x = self.b2(x, training=training)
        x = self.a3(x)
        x = self.c3(x)
        x = self.a4(x)
        return x


class Discriminator(Model):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.c1 = Conv2D(64, 4, 2, padding='valid')
        self.a1 = Activation(tf.nn.leaky_relu)

        self.c2 = Conv2D(128, 3, 2, padding='valid')
        self.b1 = BatchNormalization()
        self.a2 = Activation(tf.nn.leaky_relu)

        self.f1 = Flatten()
        self.d1 = Dense(1)

    def call(self, inputs, training=None, mask=None):
        x = self.c1(inputs)
        x = self.a1(x)
        x = self.c2(x)
        x = self.b1(x, training=training)
        x = self.a2(x)
        x = self.f1(x)
        logits = self.d1(x)
        return logits


def main():
    fashion = tf.keras.datasets.fashion_mnist
    (x_train, y_train), (x_test, y_test) = fashion.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = x_train * 2 - 1
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_train = x_train.astype('float32')

    z_dim = 100
    epochs = 100
    batch_size = 128
    learning_rate = 0.0002
    is_training = True

    train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)

    generator = Generator()
    generator.build(input_shape=(None, z_dim))
    discriminator = Discriminator()
    discriminator.build(input_shape=(None, 28, 28, 1))

    if os.path.exists('./GAN2/generatorweights/gweights.index'):
        print('-------------load the model-----------------')
        generator.load_weights('./GAN2/generatorweights/gweights')
        discriminator.load_weights('./GAN2/discriminatorweights/dweights')

    g_optimizer = tf.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)
    d_optimizer = tf.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)

    for epoch in range(epochs):
        start1 = time.perf_counter()
        for step, (x_train, y_train) in enumerate(train_db):
            batch_z = tf.random.uniform([batch_size, z_dim], minval=-1., maxval=1.)

            with tf.GradientTape() as tape:
                d_loss = d_loss_fn(generator, discriminator, batch_z, x_train, is_training)
            grads = tape.gradient(d_loss, discriminator.trainable_variables)
            d_optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))

            with tf.GradientTape() as tape:
                g_loss = g_loss_fn(generator, discriminator, batch_z, is_training)
            grads = tape.gradient(g_loss, generator.trainable_variables)
            g_optimizer.apply_gradients(zip(grads, generator.trainable_variables))

        end1 = time.perf_counter()
        print(epoch, 'd-loss:', float(d_loss), 'g-loss:', float(g_loss), end1 - start1, 's')

    generator.save_weights('./GAN2/generatorweights/gweights')
    discriminator.save_weights('./GAN2/discriminatorweights/dweights')

    f, a = plt.subplots(4, 10, figsize=(10, 4))
    for i in range(10):
        # 随机生成噪声,噪声也是均匀分布中随机取值,用训练好的G网络生成图片
        z = tf.random.uniform(shape=[4, z_dim], maxval=1, minval=-1)
        g_sample = generator(z, training=False)
        g_sample = (g_sample + 1) / 2
        g_sample = g_sample * 255.0
        g_sample = 255.0 - g_sample
        g_sample = np.reshape(g_sample, newshape=(4, 28, 28))
        # 把40张图片画出来
        for j in range(4):
            a[j][i].imshow(g_sample[j], cmap='gray')

    f.show()
    plt.draw()
    plt.waitforbuttonpress()


if __name__ == '__main__':
    main()
