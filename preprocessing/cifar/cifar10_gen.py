import sys

# sys.path.append('/disk1/hjc/classification/')  # 205
sys.path.append('/workspace/classification/code/')  # zjlab
import numpy as np
from PIL import Image
import pickle
import os
from configs import config


def unpickle(paths, types):
    CHANNEL = 3
    WIDTH = 32
    HEIGHT = 32

    data = []
    labels = []
    filenames = []
    classification = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    check_path(classification, types)

    for path in paths:
        path = '{}/{}'.format(config.datasets_CIFAR_10, path)
        with open(path, mode='rb') as file:
            # 数据集在当脚本前文件夹下
            data_dict = pickle.load(file, encoding='bytes')
            data += list(data_dict[b'data'])
            labels += list(data_dict[b'labels'])
            filenames += list(data_dict[b'filenames'])

    img = np.reshape(data, [-1, CHANNEL, WIDTH, HEIGHT])

    for i in range(img.shape[0]):
        r = img[i][0]
        g = img[i][1]
        b = img[i][2]

        ir = Image.fromarray(r)
        ig = Image.fromarray(g)
        ib = Image.fromarray(b)
        rgb = Image.merge("RGB", (ir, ig, ib))

        filename = '{}/{}/{}/{}.png'.format(config.data_cifar10, types, classification[labels[i]], i)
        rgb.save(filename, "PNG")


def check_path(classification, types):
    for cls in classification:
        data_path = '{}/{}/{}'.format(config.data_cifar10, types, cls)
        if not os.path.exists(data_path):
            os.makedirs(data_path)


def main():
    train_paths = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
    test_paths = ['test_batch']
    unpickle(train_paths, types='train')
    unpickle(test_paths, types='test')


def _test():
    import pickle
    with open("{}/data_batch_1".format(config.datasets_CIFAR_10), 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    print(dict)


if __name__ == '__main__':
    main()
    # _test()
