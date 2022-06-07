from loaders.cifar10_loader import load_images as load_cifar10
from loaders.cifar100_loader import load_images as load_cifar100
from loaders.mnist_loader import load_images as load_mnist
from loaders.fashion_mnist_loader import load_images as load_fashion_mnist
from loaders.svhn_loader import load_images as load_svhn
from loaders.stl10_loader import load_images as load_stl10
from loaders.stl10_loader import load_images_masks as load_stl10_masks
from loaders.mnin_loader import load_images as load_mnin
from loaders.mnin_loader import load_images_masks as load_mnin_masks


def load_data(data_name, data_path, data_type=None):
    print('-' * 50)
    print('DATA NAME:', data_name)
    print('DATA PATH:', data_path)
    print('DATA TYPE:', data_type)
    print('-' * 50)

    assert data_name in ['cifar10', 'cifar100', 'mnist', 'fashion_mnist', 'svhn', 'stl10', 'mnin']

    data_loader = None
    if data_name == 'cifar10':
        data_loader = load_cifar10(data_path, data_type)
    elif data_name == 'cifar100':
        data_loader = load_cifar100(data_path, data_type)
    elif data_name == 'mnist':
        data_loader = load_mnist(data_path, data_type)
    elif data_name == 'fashion_mnist':
        data_loader = load_fashion_mnist(data_path, data_type)
    elif data_name == 'svhn':
        data_loader = load_svhn(data_path, data_type)
    elif data_name == 'stl10':
        data_loader = load_stl10(data_path, data_type)
    elif data_name == 'mnin':
        data_loader = load_mnin(data_path, data_type)
    return data_loader


def load_data_mask(data_name, data_path, mask_path, data_type=None):
    print('-' * 50)
    print('DATA NAME:', data_name)
    print('DATA PATH:', data_path)
    print('DATA TYPE:', data_type)
    print('-' * 50)

    assert data_name in ['stl10', 'mnin']

    data_loader = None
    if data_name == 'stl10':
        data_loader = load_stl10_masks(data_path, mask_path, data_type)
    elif data_name == 'mnin':
        data_loader = load_mnin_masks(data_path, mask_path, data_type)
    return data_loader
