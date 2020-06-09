import tensorflow as tf
from PIL import Image
import numpy as np
import os

txtpath = './img_celeba.7z/img_celeba.7z/list_bbox_celeba.txt'

f = open(txtpath, 'r')
contents = f.readlines()
f.close()
p_path = []
x_1 = []
y_1 = []
width = []
height = []
n = 0
for content in contents:
    if n > 1:
        value = content.split()
        p_path.append(value[0])
        x_1.append(value[1])
        y_1.append(value[2])
        width.append(value[3])
        height.append(value[4])
    n += 1

x_1 = np.array(x_1, dtype=np.int)
y_1 = np.array(y_1, dtype=np.int)
width = np.array(width, dtype=np.int)
height = np.array(height, dtype=np.int)

np.save('./face/p_path.npy', p_path)
np.save('./face/x_1.npy', x_1)
np.save('./face/y_1.npy', y_1)
np.save('./face/width.npy', width)
np.save('./face/height.npy', height)

print(x_1[0:10])
