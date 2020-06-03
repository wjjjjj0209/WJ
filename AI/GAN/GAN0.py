import tensorflow as tf
import numpy as np
import time
import joblib
from matplotlib import pyplot as plt

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = np.reshape(x_train, [60000, 784])
x_train = tf.cast(x_train, dtype=tf.float32)

n_g = 0
n_d = 0
epoch = 10
batch_size = 128
lrd = 0.01
lrg = 0.02
image_dim = 784
train_loss_generate = []
train_loss_discriminate = []
loss_all_g = 0
loss_all_d = 0

gen_hidden_dim = 256
disc_hidden_dim = 256
noise_dim = 100

train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)


def xavier_init(shape):
    return tf.random.normal(shape=shape, stddev=1. / tf.sqrt(shape[0] / 2.))


# wg1 = tf.Variable(xavier_init([noise_dim, gen_hidden_dim]))
# wg2 = tf.Variable(xavier_init([gen_hidden_dim, image_dim]))
# bg1 = tf.Variable(tf.zeros([gen_hidden_dim]))
# bg2 = tf.Variable(tf.zeros([image_dim]))
#
# wd1 = tf.Variable(xavier_init([image_dim, disc_hidden_dim]))
# wd2 = tf.Variable(xavier_init([disc_hidden_dim, 1]))
# bd1 = tf.Variable(tf.zeros([disc_hidden_dim]))
# bd2 = tf.Variable(tf.zeros([1]))

[wg1, wg2, bg1, bg2, wd1, wd2, bd1, bd2] = joblib.load('parameter.pkl')


def generate(x):
    hidden_layer = tf.nn.leaky_relu(tf.add(tf.matmul(x, wg1), bg1))
    out_layer = tf.nn.tanh(tf.add(tf.matmul(hidden_layer, wg2), bg2))
    return out_layer


def discriminator(x):
    hidden_layer = tf.nn.leaky_relu(tf.add(tf.matmul(x, wd1), bd1))
    out_layer = tf.nn.sigmoid(tf.add(tf.matmul(hidden_layer, wd2), bd2))
    return out_layer


for epoch in range(epoch):
    start1 = time.perf_counter()
    for step, (x_train, y_train) in enumerate(train_db):
        z_input = tf.random.uniform(shape=[x_train.shape[0], noise_dim], minval=-1, maxval=1)
        with tf.GradientTape() as tape:
            gen_sample = generate(z_input)
            disc_fake = discriminator(gen_sample)
            disc_real = discriminator(x_train)
            disc_loss = -tf.reduce_mean(tf.math.log(disc_real) + tf.math.log(1. - disc_fake))
            loss_all_d += disc_loss.numpy()

        grads_d = tape.gradient(disc_loss, [wd1, wd2, bd1, bd2])

        wd1.assign_sub(lrd * grads_d[0])
        wd2.assign_sub(lrd * grads_d[1])
        bd1.assign_sub(lrd * grads_d[2])
        bd2.assign_sub(lrd * grads_d[3])

        if step % 5 == 0:
            with tf.GradientTape() as tape:
                gen_sample = generate(z_input)
                disc_fake = discriminator(gen_sample)
                gen_loss = -tf.reduce_mean(tf.math.log(disc_fake))
                loss_all_g += gen_loss.numpy()

            grads_g = tape.gradient(gen_loss, [wg1, wg2, bg1, bg2])

            wg1.assign_sub(lrg * grads_g[0])
            wg2.assign_sub(lrg * grads_g[1])
            bg1.assign_sub(lrg * grads_g[2])
            bg2.assign_sub(lrg * grads_g[3])

            n_g += 1

        n_d = step + 1

    end1 = time.perf_counter()
    print("Epoch {}, generate_loss: {}, discriminate_loss: {}, timecost: {}s".format(epoch, loss_all_g / n_g,
                                                                                     loss_all_d / n_d, end1-start1))
    train_loss_generate.append(loss_all_g / n_g)
    train_loss_discriminate.append(loss_all_d / n_d)
    loss_all_g = 0
    loss_all_d = 0
    n_g = 0

joblib.dump([wg1, wg2, bg1, bg2, wd1, wd2, bd1, bd2], 'parameter.pkl')

plt.title('Loss Function Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(train_loss_generate, label="$generateloss$")
plt.plot(train_loss_discriminate, label="$discriminateloss$")
plt.legend()
plt.show()

f, a = plt.subplots(4, 10, figsize=(10, 4))
# 生成4张x10轮图片,共40张图片
for i in range(10):
    # 随机生成噪声,噪声也是均匀分布中随机取值,用训练好的G网络生成图片
    z_input = tf.random.uniform(shape=[4, noise_dim], maxval=1, minval=-1)
    g_sample = generate(z_input)
    g_sample = g_sample * 255.0
    g_sample = 255.0 - g_sample
    g_sample = np.reshape(g_sample, newshape=(4, 28, 28))
    # 把40张图片画出来
    for j in range(4):
        a[j][i].imshow(g_sample[j], cmap='gray')

f.show()
plt.draw()
plt.waitforbuttonpress()
