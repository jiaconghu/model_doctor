import random
from collections.abc import Sequence, Iterable
from torchvision import transforms
import torchvision.transforms.functional as F


class Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, images):
        for t in self.transforms:
            images = t(images)
        return images


class ToTensor(object):

    def __call__(self, images):
        trans = []
        # TODO mask to tensor?
        for img in images:
            img = F.to_tensor(img)
            trans.append(img)
        return trans


class Normalize(object):
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, tensors):
        norms = [F.normalize(tensors[0], self.mean, self.std, self.inplace), tensors[1]]
        return norms

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class Resize(object):
    # def __init__(self, size, interpolation=2):
    def __init__(self, size, interpolation=transforms.InterpolationMode.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, images):
        trans = []
        for img in images:
            img = F.resize(img, self.size, self.interpolation)
            trans.append(img)
        return trans


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, images):
        if random.random() < self.p:
            trans = []
            for img in images:
                img = F.hflip(img)
                trans.append(img)
            return trans
        return images


class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, images):
        if random.random() < self.p:
            trans = []
            for img in images:
                img = F.vflip(img)
                trans.append(img)
            return trans
        return images


class RandomRotation(transforms.RandomRotation):

    def __init__(self, degrees):
        super(RandomRotation, self).__init__(degrees)

    def __call__(self, images):
        angle = self.get_params(self.degrees)
        trans = []
        for img in images:
            img = F.rotate(img, angle, self.resample, self.expand, self.center, self.fill)
            trans.append(img)
        return trans


class RandomResizedCrop(transforms.RandomResizedCrop):
    def __init__(self, size, scale=(0.08, 1.0)):
        super(RandomResizedCrop, self).__init__(size, scale)

    def __call__(self, images):
        i, j, h, w = self.get_params(images[0], self.scale, self.ratio)
        trans = []
        for img in images:
            img = F.resized_crop(img, i, j, h, w, self.size, self.interpolation)
            trans.append(img)
        return trans


class RandomCrop(transforms.RandomCrop):
    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode="constant"):
        super().__init__(size, padding, pad_if_needed, fill, padding_mode)

    def forward(self, images):
        if self.padding is not None:
            for i, img in enumerate(images):
                images[i] = F.pad(img, self.padding, self.fill, self.padding_mode)

        width, height = F._get_image_size(images[0])
        # pad the width if needed
        if self.pad_if_needed and width < self.size[1]:
            padding = [self.size[1] - width, 0]

            for i, img in enumerate(images):
                images[i] = F.pad(img, padding, self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and height < self.size[0]:
            padding = [0, self.size[0] - height]
            for i, img in enumerate(images):
                images[i] = F.pad(img, padding, self.fill, self.padding_mode)

        i, j, h, w = self.get_params(images[0], self.size)

        for i, img in enumerate(images):
            images[i] = F.crop(img, i, j, h, w)
        return images
