import os
import time

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Conv2DTranspose, MaxPool2D,\
    BatchNormalization, Dropout, Activation

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def reparameterize(mean, logvar):
    eps = tf.random.normal(shape=mean.shape)
    return eps * tf.exp(logvar * 0.5) + mean


class Vae(Model):
    def __init__(self):
        super(Vae, self).__init__()

        self.ec1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')
        self.eb1 = BatchNormalization()
        self.ea1 = Activation('relu')
        self.ep1 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
        self.ed1 = Dropout(0.2)

        self.ec2 = Conv2D(filters=128, kernel_size=(3, 3), padding='same')
        self.eb2 = BatchNormalization()
        self.ea2 = Activation('relu')
        self.ep2 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
        self.ed2 = Dropout(0.2)

        self.ec3 = Conv2D(filters=256, kernel_size=(3, 3), padding='same')
        self.eb3 = BatchNormalization()
        self.ea3 = Activation('relu')
        self.ec4 = Conv2D(filters=256, kernel_size=(3, 3), padding='same')
        self.eb4 = BatchNormalization()
        self.ea4 = Activation('relu')
        self.ep3 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
        self.ed3 = Dropout(0.2)

        self.ec5 = Conv2D(filters=512, kernel_size=(3, 3), padding='same')
        self.eb5 = BatchNormalization()
        self.ea5 = Activation('relu')
        self.ec6 = Conv2D(filters=512, kernel_size=(3, 3), padding='same')
        self.eb6 = BatchNormalization()
        self.ea6 = Activation('relu')
        self.ep4 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
        self.ed4 = Dropout(0.2)

        self.ef1 = Flatten()
        self.ef2 = Dense(256, activation='relu')
        self.edu = Dense(100)
        self.edv = Dense(100)
        # 解码层与编码层分解--------------------------------------------------

        self.df1 = Dense(256, activation='relu')
        self.df2 = Dense(2 * 2 * 512, activation='relu')

        self.dc1 = Conv2DTranspose(512, 3, 2, padding='SAME')
        self.db1 = BatchNormalization()
        self.da1 = Activation('relu')
        self.dc2 = Conv2DTranspose(512, 3, 1, padding='SAME')
        self.db2 = BatchNormalization()
        self.da2 = Activation('relu')
        self.dd1 = Dropout(0.2)

        self.dc3 = Conv2DTranspose(256, 3, 2, padding='SAME')
        self.db3 = BatchNormalization()
        self.da3 = Activation('relu')
        self.dc4 = Conv2DTranspose(256, 3, 1, padding='SAME')
        self.db4 = BatchNormalization()
        self.da4 = Activation('relu')
        self.dd2 = Dropout(0.2)

        self.dc5 = Conv2DTranspose(128, 3, 2, padding='SAME')
        self.db5 = BatchNormalization()
        self.da5 = Activation('relu')
        self.dd3 = Dropout(0.2)

        self.dc6 = Conv2DTranspose(64, 3, 2, padding='SAME')
        self.db6 = BatchNormalization()
        self.da6 = Activation('relu')
        self.dd4 = Dropout(0.2)

        self.dc7 = Conv2DTranspose(32, 3, 1, padding='SAME')
        self.db7 = BatchNormalization()
        self.da7 = Activation('relu')

        self.dc8 = Conv2DTranspose(3, 3, 1, padding='SAME')

    def encode(self, x):
        x = self.ec1(x)
        x = self.eb1(x)
        x = self.ea1(x)
        x = self.ep1(x)
        x = self.ed1(x)

        x = self.ec2(x)
        x = self.eb2(x)
        x = self.ea2(x)
        x = self.ep2(x)
        x = self.ed2(x)

        x = self.ec3(x)
        x = self.eb3(x)
        x = self.ea3(x)
        x = self.ec4(x)
        x = self.eb4(x)
        x = self.ea4(x)
        x = self.ep3(x)
        x = self.ed3(x)

        x = self.ec5(x)
        x = self.eb5(x)
        x = self.ea5(x)
        x = self.ec6(x)
        x = self.eb6(x)
        x = self.ea6(x)
        x = self.ep4(x)
        x = self.ed4(x)

        x = self.ef1(x)
        x = self.ef2(x)
        mean = self.edu(x)
        logvar = self.edv(x)
        return mean, logvar

    def decode(self, z):
        z = self.df1(z)
        z = self.df2(z)
        z = tf.reshape(z, [-1, 2, 2, 512])

        z = self.dc1(z)
        z = self.db1(z)
        z = self.da1(z)
        z = self.dc2(z)
        z = self.db2(z)
        z = self.da2(z)
        z = self.dd1(z)

        z = self.dc3(z)
        z = self.db3(z)
        z = self.da3(z)
        z = self.dc4(z)
        z = self.db4(z)
        z = self.da4(z)
        z = self.dd2(z)

        z = self.dc5(z)
        z = self.db5(z)
        z = self.da5(z)
        z = self.dd3(z)

        z = self.dc6(z)
        z = self.db6(z)
        z = self.da6(z)
        z = self.dd4(z)

        z = self.dc7(z)
        z = self.db7(z)
        z = self.da7(z)

        logits = self.dc8(z)
        return logits


def gsnoise(img, mean=0, var=0.001):
    noise = np.random.normal(mean, var ** 0.5, size=img.shape)
    out = img + noise
    out = np.clip(out, 0., 1.)
    return out


def main():
    epochs = 50
    lr = 0.001
    batch_size = 128

    cifar10 = tf.keras.datasets.cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = x_train.reshape(x_train.shape[0], 32, 32, 3)
    x_train = x_train.astype('float32')
    x_train_g = gsnoise(x_train)
    x_train_g = x_train_g.astype('float32')
    x_test = x_test.reshape(x_test.shape[0], 32, 32, 3)
    x_test = x_test.astype('float32')
    x_test_g = gsnoise(x_test)
    x_test_g = x_test_g.astype('float32')

    train_db = tf.data.Dataset.from_tensor_slices((x_train_g, x_train)).batch(batch_size)

    model = Vae()
    optimizer = tf.keras.optimizers.Adam(lr)
    if os.path.exists('./vae4/vae.index'):
        print('-------------load the model-----------------')
        model.load_weights('./vae4/vae')

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

    model.save_weights('./vae4/vae')

    f, a = plt.subplots(3, 10, figsize=(10, 3))

    randindex = np.random.randint(0, len(x_test_g), 10)
    initx = x_test_g[randindex]
    realx = x_test[randindex]

    mean, logvar = model.encode(initx)
    z = reparameterize(mean, logvar)
    x_logits = model.decode(z)
    outputx = tf.nn.sigmoid(x_logits)

    initx = initx * 255.0
    outputx = outputx * 255.0
    realx = realx * 255.0
    initx = np.reshape(initx, newshape=[10, 32, 32, 3]).astype(np.int)
    outputx = np.reshape(outputx, newshape=[10, 32, 32, 3]).astype(np.int)
    realx = np.reshape(realx, newshape=[10, 32, 32, 3]).astype(np.int)

    for i in range(10):
        a[0][i].imshow(initx[i])
        a[0][i].axis('off')
        a[1][i].imshow(outputx[i])
        a[1][i].axis('off')
        a[2][i].imshow(realx[i])
        a[2][i].axis('off')

    f.show()
    plt.draw()
    plt.waitforbuttonpress()


if __name__ == '__main__':
    main()
