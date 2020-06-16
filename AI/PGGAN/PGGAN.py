import os
import time

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

import PGmodel
import config

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def train(train_db, level, generator, discriminator, g_optimizer, d_optimizer, transit):
    for epoch in range(config.epochs):
        start1 = time.perf_counter()
        for step, x_train in enumerate(train_db):

            for i in range(6 - level):
                x_train = PGmodel.downsampling2d(x_train)

            z_input = tf.random.uniform([x_train.shape[0], config.z_dim], minval=-1., maxval=1.)
            trans_alpha = step / int(config.epochs * config.data_size / config.batch_size)

            with tf.GradientTape() as tape:
                fake_img = generator.generate(z_input, level, transit, trans_alpha)
                g_logits = discriminator.discriminate(fake_img, level, transit, trans_alpha)
                d_logits = discriminator.discriminate(x_train, level, transit, trans_alpha)
                d_loss = tf.reduce_mean(g_logits - d_logits)
                # 梯度惩罚
                differences = fake_img - x_train
                alpha = tf.random.uniform(shape=[x_train.shape[0], 1, 1, 1], minval=0., maxval=1.)
                interpolates = x_train + (alpha * differences)
                with tf.GradientTape() as tape1:
                    tape1.watch(interpolates)
                    discri_logits = discriminator.discriminate(interpolates, level, transit, trans_alpha)
                gradients = tape1.gradient(discri_logits, [interpolates, ])[0]
                slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
                gradient_penalty = tf.reduce_mean(tf.square((slopes - 1.)))

                d_loss += config.lam_gp * gradient_penalty
                d_loss += config.lam_eps * tf.reduce_mean(tf.square(d_logits))
            grads = tape.gradient(d_loss, discriminator.trainable_variables)
            d_optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))

            with tf.GradientTape() as tape:
                fake_img = generator.generate(z_input, level, transit, trans_alpha)
                g_logits = discriminator.discriminate(fake_img, level, transit, trans_alpha)
                g_loss = - tf.reduce_mean(g_logits)
            grads = tape.gradient(g_loss, generator.trainable_variables)
            g_optimizer.apply_gradients(zip(grads, generator.trainable_variables))

        end1 = time.perf_counter()
        print(epoch, 'd-loss:', float(d_loss), 'g-loss:', float(g_loss), end1 - start1, 's')


def main():
    x_train = np.load('./facedata64.npy')
    x_train = x_train / 255.
    x_train = x_train * 2 - 1
    x_train = x_train.astype('float32')
    train_db = tf.data.Dataset.from_tensor_slices(x_train).batch(config.batch_size)

    generator = PGmodel.Generator(config.levels)
    discriminator = PGmodel.Discriminator(config.levels)
    g_optimizer = tf.optimizers.Adam(learning_rate=config.learning_rate, beta_1=config.beta1,
                                     beta_2=config.beta2, epsilon=config.ada_ep)
    d_optimizer = tf.optimizers.Adam(learning_rate=config.learning_rate, beta_1=config.beta1,
                                     beta_2=config.beta2, epsilon=config.ada_ep)

    checkpoint_dir = 'PGGAN'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(g_optimizer=g_optimizer,
                                     d_optimizer=d_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)

    if os.path.exists('PGGAN'):
        print('-------------load the model-----------------')
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    for level in config.levels:
        if level == 2:
            print(level)
            generator.rgb[level - 2].trainable = True
            discriminator.rgb[level - 2].trainable = True
            train(train_db, level, generator, discriminator, g_optimizer, d_optimizer, transit=False)
            generator.rgb[level - 2].trainable = False
            discriminator.rgb[level - 2].trainable = False
        else:
            print(level)
            generator.rgb[level - 2].trainable = True
            discriminator.rgb[level - 2].trainable = True
            generator.rgb[level - 3].trainable = True
            discriminator.rgb[level - 3].trainable = True
            train(train_db, level, generator, discriminator, g_optimizer, d_optimizer, transit=True)
            generator.rgb[level - 3].trainable = False
            discriminator.rgb[level - 3].trainable = False
            train(train_db, level, generator, discriminator, g_optimizer, d_optimizer, transit=False)
            generator.rgb[level - 2].trainable = False
            discriminator.rgb[level - 2].trainable = False

        checkpoint.save(file_prefix=checkpoint_prefix)

    f, a = plt.subplots(4, 5, figsize=(6, 6))
    for i in range(4):
        # 随机生成噪声
        z = tf.random.uniform(shape=[5, config.z_dim], maxval=1, minval=-1)
        level = 5
        g_sample = generator.generate(z, level=level, transit=False, trans_alpha=1)
        g_sample = (g_sample + 1) / 2
        g_sample = np.reshape(g_sample, newshape=(5, 2 ** level, 2 ** level, 3))
        g_sample = np.clip(g_sample, 0, 1)
        # 画图
        for j in range(5):
            a[i][j].imshow(g_sample[j])
            a[i][j].axis('off')

    plt.show()


if __name__ == '__main__':
    main()
