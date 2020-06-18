import os
import time

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

import SGmodel

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def main():

    epochs = 1
    batch_size = 16
    learning_rate = 0.0001
    beta1 = 0.5
    beta2 = 0.999
    cls_weight = 1
    rec_weight = 10
    lam_gp = 10
    n_crtic = 5
    x_train = np.load('facedata64.npy')
    label_train = np.load('labelvalue.npy')

    x_train = x_train / 127.5 - 1
    x_train = x_train.astype('float32')
    label_train = label_train.astype('float32')

    train_db = tf.data.Dataset.from_tensor_slices((x_train, label_train)).batch(batch_size)

    generator = SGmodel.Generator()
    discriminator = SGmodel.Discriminator()

    g_optimizer = tf.optimizers.Adam(learning_rate=learning_rate, beta_1=beta1, beta_2=beta2)
    d_optimizer = tf.optimizers.Adam(learning_rate=learning_rate, beta_1=beta1, beta_2=beta2)

    checkpoint_dir = './StarGAN'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(g_optimizer=g_optimizer,
                                     d_optimizer=d_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)

    if os.path.exists('./StarGAN'):
        print('-------------load the model-----------------')
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    for epoch in range(epochs):
        start1 = time.perf_counter()
        for step, (inputimg, label0) in enumerate(train_db):
            label1 = tf.random.shuffle(label0)

            with tf.GradientTape() as tape:
                fake_img = generator.generate(inputimg, label1)

                real_logits, real_cls = discriminator.discriminate(inputimg)
                fake_logits, fake_cls = discriminator.discriminate(fake_img)

                d_adv_loss = tf.reduce_mean(fake_logits - real_logits)
                d_cls_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label0, logits=real_cls))

                differences = fake_img - inputimg
                alpha = tf.random.uniform(shape=[inputimg.shape[0], 1, 1, 1], minval=0., maxval=1.)
                interpolates = inputimg + (alpha * differences)
                with tf.GradientTape() as tape1:
                    tape1.watch(interpolates)
                    discri_logits, _ = discriminator.discriminate(interpolates)
                gradients = tape1.gradient(discri_logits, [interpolates, ])[0]
                slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
                gradient_penalty = tf.reduce_mean(tf.square((slopes - 1.)))
                d_adv_loss += lam_gp * gradient_penalty
                d_loss = d_adv_loss + cls_weight * d_cls_loss
            grads = tape.gradient(d_loss, discriminator.trainable_variables)
            d_optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))

            if step % n_crtic == 0:
                with tf.GradientTape() as tape:
                    fake_img = generator.generate(inputimg, label1)
                    rec_img = generator.generate(fake_img, label0)

                    fake_logits, fake_cls = discriminator.discriminate(fake_img)

                    g_adv_loss = -tf.reduce_mean(fake_logits)
                    g_cls_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label1, logits=fake_cls))
                    g_rec_loss = tf.reduce_mean(tf.abs(inputimg - rec_img))
                    g_loss = g_adv_loss + cls_weight * g_cls_loss + rec_weight * g_rec_loss
                grads = tape.gradient(g_loss, generator.trainable_variables)
                g_optimizer.apply_gradients(zip(grads, generator.trainable_variables))

        end1 = time.perf_counter()
        print(epoch, 'd-loss:', float(d_loss), 'g-loss:', float(g_loss), end1 - start1, 's')

    checkpoint.save(file_prefix=checkpoint_prefix)

    f, a = plt.subplots(4, 5, figsize=(6, 6))
    label_fix = np.load('labelfix.npy')
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


if __name__ == '__main__':
    main()
