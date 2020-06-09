import tensorflow as tf
from PIL import Image
import numpy as np
import os

p_path = np.load('./face/p_path.npy')
x_1 = np.load('./face/x_1.npy')
y_1 = np.load('./face/y_1.npy')
width = np.load('./face/width.npy')
height = np.load('./face/height.npy')

for i in range(10000):
    img = Image.open('./img_celeba.7z/img_celeba.7z/img_celeba/{}'.format(p_path[i]))
    img_face = img.crop((x_1[i], y_1[i], x_1[i] + width[i], y_1[i] + height[i]))
    img_face.save('./face/face/face{}'.format(p_path[i]))
