import os

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

import GDmodel
import config

generator = GDmodel.Generator(config.levels)
discriminator = GDmodel.Discriminator(config.levels)
g_optimizer = tf.optimizers.Adam(learning_rate=config.learning_rate, beta_1=config.beta1, beta_2=config.beta2)
d_optimizer = tf.optimizers.Adam(learning_rate=config.learning_rate, beta_1=config.beta1, beta_2=config.beta2)

checkpoint_dir = './PGGAN'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(g_optimizer=g_optimizer,
                                 d_optimizer=d_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

f, a = plt.subplots(4, 5, figsize=(6, 6))
for i in range(5):
    # 随机生成噪声
    z = tf.random.uniform(shape=[4, config.z_dim], maxval=1, minval=-1)
    level = 4
    g_sample = generator.gerenate(z, level=level, transit=False, trans_alpha=1)
    g_sample = (g_sample + 1) / 2
    g_sample = np.reshape(g_sample, newshape=(4, 2 ** level, 2 ** level, 3))
    g_sample = np.clip(g_sample, 0, 1)
    # 画图
    for j in range(4):
        a[j][i].imshow(g_sample[j])
        a[j][i].axis('off')

plt.show()
