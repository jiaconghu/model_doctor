import os
import PIL.Image as Image
from torch.utils.data import Dataset


def _img_loader(path, mode='RGB', default_mask_path=None):
    assert mode in ['RGB', 'L']
    # --------------------------------------
    if not os.path.exists(path):
        path = default_mask_path
    # --------------------------------------
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert(mode)


def _find_classes(root):
    class_names = [d.name for d in os.scandir(root) if d.is_dir()]
    class_names.sort()
    classes_indices = {class_names[i]: i for i in range(len(class_names))}
    # print(classes_indices)
    return class_names, classes_indices  # 'class_name':index


def _make_dataset(image_dir, mask_dir):
    samples = []  # image_path, mask_path, class_idx

    class_names, class_indices = _find_classes(image_dir)

    for class_name in sorted(class_names):
        class_idx = class_indices[class_name]
        target_dir = os.path.join(image_dir, class_name)

        if not os.path.isdir(target_dir):
            continue

        for root, _, files in sorted(os.walk(target_dir)):
            for file in sorted(files):
                image_path = os.path.join(root, file)
                mask_path = os.path.join(mask_dir, file.replace('jpg', 'png'))
                item = image_path, mask_path, class_idx
                samples.append(item)
    return samples


class ImageMaskDataset(Dataset):
    def __init__(self, image_dir, mask_dir, default_mask_path, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.default_mask_path = default_mask_path
        self.transform = transform
        self.samples = _make_dataset(self.image_dir, self.mask_dir)
        self.targets = [s[2] for s in self.samples]

    def __getitem__(self, index):
        image_path, mask_path, target = self.samples[index]
        image = _img_loader(image_path, mode='RGB')
        mask = _img_loader(mask_path, mode='L', default_mask_path=self.default_mask_path)

        images = [image, mask]
        if self.transform is not None:
            images = self.transform(images)

        return images[0], target, images[1]

    def __len__(self):
        return len(self.samples)
