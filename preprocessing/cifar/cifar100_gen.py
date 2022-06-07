import sys

sys.path.append('/workspace/classification/code/')  # zjlab
import os
import pickle
import cv2
import numpy as np

from configs import config
from utils import image_util
from PIL import Image

# source directory
CIFAR100_DIR = config.datasets_CIFAR_100

# extract cifar img in here.
CIFAR100_TRAIN_DIR = config.data_cifar100 + '/train'
CIFAR100_TEST_DIR = config.data_cifar100 + '/test'

dir_list = [CIFAR100_TRAIN_DIR, CIFAR100_TEST_DIR]


# extract the binaries, encoding must is 'bytes'!
def unpickle(file):
    fo = open(file, 'rb')
    data = pickle.load(fo, encoding='bytes')
    fo.close()
    return data


def gen_cifar_100():
    # generate training data sets.

    data_dir = CIFAR100_DIR + '/train'
    train_data = unpickle(data_dir)
    print(data_dir + " is loading...")

    for i in range(0, 50000):
        # binary files are converted to images.
        img = np.reshape(train_data[b'data'][i], (3, 32, 32))
        # img = img.transpose(1, 2, 0)
        # img_path = CIFAR100_TRAIN_DIR + '/' + str(train_data[b'fine_labels'][i]) + '/' + str(i) + '.jpg'
        # print(img_path)
        # image_util.save_cv(img, img_path)

        r = img[0]
        g = img[1]
        b = img[2]

        ir = Image.fromarray(r)
        ig = Image.fromarray(g)
        ib = Image.fromarray(b)
        rgb = Image.merge("RGB", (ir, ig, ib))
        img_path = CIFAR100_TRAIN_DIR + '/' + str(train_data[b'fine_labels'][i])
        if not os.path.exists(img_path):
            os.makedirs(img_path)
        filename = img_path + '/' + str(i) + '.png'
        print(filename)
        rgb.save(filename, "PNG")
    print(data_dir + " loaded.")

    print("test_batch is loading...")

    # generate the validation data set.
    val_data = CIFAR100_DIR + '/test'
    val_data = unpickle(val_data)
    for i in range(0, 10000):
        # binary files are converted to images
        img = np.reshape(val_data[b'data'][i], (3, 32, 32))
        # img = img.transpose(1, 2, 0)
        # img_path = CIFAR100_TEST_DIR + '/' + str(val_data[b'fine_labels'][i]) + '/' + str(i) + '.jpg'
        # print(img_path)
        # image_util.save_cv(img, img_path)

        r = img[0]
        g = img[1]
        b = img[2]

        ir = Image.fromarray(r)
        ig = Image.fromarray(g)
        ib = Image.fromarray(b)
        rgb = Image.merge("RGB", (ir, ig, ib))
        img_path = CIFAR100_TEST_DIR + '/' + str(train_data[b'fine_labels'][i])
        if not os.path.exists(img_path):
            os.makedirs(img_path)
        filename = img_path + '/' + str(i) + '.png'
        print(filename)
        rgb.save(filename, "PNG")
    print("test_batch loaded.")
    return


if __name__ == '__main__':
    gen_cifar_100()
