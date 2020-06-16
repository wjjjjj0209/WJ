import os

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

import SGmodel

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

x_train = np.load('facedata64.npy')

x_train = x_train / 127.5 - 1
x_train = x_train.astype('float32')
label_fix = np.load('labelfix.npy')

generator = SGmodel.Generator()
discriminator = SGmodel.Discriminator()
g_optimizer = tf.optimizers.Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.999)
d_optimizer = tf.optimizers.Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.999)

checkpoint_dir = 'PGGAN_pn in dis'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(g_optimizer=g_optimizer,
                                 d_optimizer=d_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

f, a = plt.subplots(4, 5, figsize=(6, 6))

for i in range(4):
    randindex = np.random.randint(0, len(x_train), 1)
    initx = x_train[randindex]
    outlabel = label_fix[randindex]
    outlabel = np.transpose(outlabel, axes=[1, 0, 2])

    initx = np.reshape(initx, newshape=[1, 64, 64, 3])
    outlabel = np.reshape(outlabel, newshape=[5, 1, 5])

    outimg = []
    for k in range(5):
        outimg.append(generator.generate(initx, outlabel[k]))
    outimg = np.array(outimg)
    outimg = outimg / 2 + 0.5
    outimg = np.reshape(outimg, newshape=[5, 64, 64, 3])
    outimg = np.clip(outimg, 0, 1)

    # 画图
    for j in range(5):
        a[i][j].imshow(outimg[j])
        a[i][j].axis('off')

plt.show()
