from torch.utils.data import DataLoader
from torchvision import transforms
from loaders.datasets import ImageDataset
from loaders.datasets import ImageMaskDataset


def _get_train_set(data_path):
    return ImageDataset(image_dir=data_path,
                        transform=transforms.Compose([
                            transforms.Resize((224, 224)),
                            transforms.RandomHorizontalFlip(0.5),
                            transforms.RandomVerticalFlip(0.5),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406],
                                                 [0.229, 0.224, 0.225])
                        ]))


def _get_test_set(data_path):
    return ImageDataset(image_dir=data_path,
                        transform=transforms.Compose([
                            transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406],
                                                 [0.229, 0.224, 0.225])
                        ]))


def _get_train_set_mask(data_path, mask_path):
    return ImageMaskDataset(image_dir=data_path,
                            mask_dir=mask_path,
                            default_mask_path='xxx',
                            transform=transforms.Compose([
                                transforms.Resize((224, 224)),
                                transforms.RandomHorizontalFlip(0.5),
                                transforms.RandomVerticalFlip(0.5),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406],
                                                     [0.229, 0.224, 0.225])
                            ]))


def load_images(data_path, data_type=None):
    assert data_type is None or data_type in ['train', 'test']
    if data_type == 'train':
        data_set = _get_train_set(data_path)
    else:
        data_set = _get_test_set(data_path)

    data_loader = DataLoader(dataset=data_set,
                             batch_size=32,
                             num_workers=4,
                             shuffle=True)

    return data_loader


def load_images_masks(data_path, mask_path, data_type=None):
    assert data_type is None or data_type in ['train']

    data_set = _get_train_set_mask(data_path, mask_path)

    data_loader = DataLoader(dataset=data_set,
                             batch_size=32,
                             num_workers=4,
                             shuffle=True)

    return data_loader
