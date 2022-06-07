import sys

sys.path.append('/workspace/classification/code/')  # zjl

import numpy as np
import cv2
import os

from configs import config
from utils import image_util


def save_mnist_to_jpg(mnist_image_file, mnist_label_file, save_dir):
    if 'train' in os.path.basename(mnist_image_file):
        num_file = 60000
        prefix = 'train'
    else:
        num_file = 10000
        prefix = 'test'
    with open(mnist_image_file, 'rb') as f1:
        image_file = f1.read()
    with open(mnist_label_file, 'rb') as f2:
        label_file = f2.read()
    image_file = image_file[16:]
    label_file = label_file[8:]
    for i in range(num_file):
        label = int(label_file[i])
        image_list = [int(item) for item in image_file[i * 784:i * 784 + 784]]
        image_np = np.array(image_list, dtype=np.uint8).reshape(28, 28)
        save_name = os.path.join(save_dir, str(label), '{}_{}.jpg'.format(prefix, i))
        # gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        image_util.save_cv(image_np, save_name)
        print('{} ==> {}_{}_{}.jpg'.format(i, prefix, i, label))


if __name__ == '__main__':
    datasets_path = config.datasets_FASHION_MNIST
    train_image_file = datasets_path + '/train-images-idx3-ubyte'
    train_label_file = datasets_path + '/train-labels-idx1-ubyte'
    test_image_file = datasets_path + '/t10k-images-idx3-ubyte'
    test_label_file = datasets_path + '/t10k-labels-idx1-ubyte'

    save_train_dir = config.data_fashion_mnist + '/train'
    save_test_dir = config.data_fashion_mnist + '/test'

    if not os.path.exists(save_train_dir):
        os.makedirs(save_train_dir)
    if not os.path.exists(save_test_dir):
        os.makedirs(save_test_dir)

    save_mnist_to_jpg(train_image_file, train_label_file, save_train_dir)
    save_mnist_to_jpg(test_image_file, test_label_file, save_test_dir)
