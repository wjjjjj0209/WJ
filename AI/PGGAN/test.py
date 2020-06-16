import os

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

import PGmodel
import config

generator = PGmodel.Generator(config.levels)
discriminator = PGmodel.Discriminator(config.levels)
g_optimizer = tf.optimizers.Adam(learning_rate=config.learning_rate, beta_1=config.beta1, beta_2=config.beta2)
d_optimizer = tf.optimizers.Adam(learning_rate=config.learning_rate, beta_1=config.beta1, beta_2=config.beta2)

checkpoint_dir = 'PGGAN'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(g_optimizer=g_optimizer,
                                 d_optimizer=d_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

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
