import os
import time

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Conv2DTranspose

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def reparameterize(mean, logvar):
    eps = tf.random.normal(shape=mean.shape)
    return eps * tf.exp(logvar * 0.5) + mean


class Vae(Model):
    def __init__(self):
        super(Vae, self).__init__()

        self.ec1 = Conv2D(filters=32, kernel_size=3, strides=2, activation='relu')
        self.ec2 = Conv2D(filters=32, kernel_size=3, strides=2, activation='relu')
        self.ef1 = Flatten()
        self.ed1 = Dense(256, activation='relu')
        self.edu = Dense(100)
        self.edv = Dense(100)

        self.dd1 = Dense(7*7*32, activation='relu')
        self.dc1 = Conv2DTranspose(64, 3, 2, padding='SAME', activation='relu')
        self.dc2 = Conv2DTranspose(32, 3, 2, padding='SAME', activation='relu')
        self.dc3 = Conv2DTranspose(1, 3, 1, padding='SAME', activation='relu')
        self.dc4 = Conv2DTranspose(1, 3, 1, padding='SAME')

    def encode(self, x):
        x = self.ec1(x)
        x = self.ec2(x)
        x = self.ef1(x)
        x = self.ed1(x)
        mean = self.edu(x)
        logvar = self.edv(x)
        return mean, logvar

    def decode(self, z):
        z = self.dd1(z)
        z = tf.reshape(z, [-1, 7, 7, 32])
        z = self.dc1(z)
        z = self.dc2(z)
        z = self.dc3(z)
        logits = self.dc4(z)
        return logits


def gsnoise(img, mean=0, var=0.1):
    noise = np.random.normal(mean, var ** 0.5, size=img.shape)
    out = img + noise
    out = np.clip(out, 0., 1.)
    return out


def main():
    epochs = 100
    lr = 0.001
    batch_size = 100

    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_train = x_train.astype('float32')
    x_train_g = gsnoise(x_train)
    x_train_g = x_train_g.astype('float32')
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    x_test = x_test.astype('float32')
    x_test_g = gsnoise(x_test)
    x_test_g = x_test_g.astype('float32')

    train_db = tf.data.Dataset.from_tensor_slices((x_train_g, x_train)).batch(batch_size)

    model = Vae()
    optimizer = tf.keras.optimizers.Adam(lr)
    if os.path.exists('./vae2/vae.index'):
        print('-------------load the model-----------------')
        model.load_weights('./vae2/vae')

    for epoch in range(epochs):
        start1 = time.perf_counter()
        for step, (x_train_g, x_train) in enumerate(train_db):
            with tf.GradientTape() as tape:
                mean, logvar = model.encode(x_train_g)
                z = reparameterize(mean, logvar)
                x_logits = model.decode(z)

                cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logits, labels=x_train)
                marginal_likelihood = - tf.reduce_sum(cross_ent, axis=[1, 2, 3])
                marginal_likelihood = tf.reduce_mean(marginal_likelihood)

                kl_divergence = tf.reduce_sum(mean ** 2 + tf.exp(logvar) - logvar - 1, axis=1)
                kl_divergence = tf.reduce_mean(kl_divergence)

                loss = -marginal_likelihood + kl_divergence

            grads = tape.gradient(loss, model.trainable_variables)

            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        last_loss = loss
        end1 = time.perf_counter()
        print(epoch, 'last-loss:', float(last_loss), end1 - start1, 's')

    model.save_weights('./vae2/vae')

    f, a = plt.subplots(2, 10, figsize=(10, 2))

    randindex = np.random.randint(0, len(x_test_g), 10)
    # initx = x_test_g[randindex]
    initx = np.random.normal()

    mean, logvar = model.encode(initx)
    z = reparameterize(mean, logvar)
    x_logits = model.decode(z)
    outputx = tf.nn.sigmoid(x_logits)

    initx = initx * 255.0
    initx = 255.0 - initx
    outputx = outputx * 255.0
    outputx = 255.0 - outputx
    initx = np.reshape(initx, newshape=[10, 28, 28])
    outputx = np.reshape(outputx, newshape=[10, 28, 28])

    for i in range(10):
        a[0][i].imshow(initx[i], cmap='gray')
        a[1][i].imshow(outputx[i], cmap='gray')

    f.show()
    plt.draw()
    plt.waitforbuttonpress()


if __name__ == '__main__':
    main()
