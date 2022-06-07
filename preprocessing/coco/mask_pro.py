import sys

# sys.path.append('/disk2/hjc/classification/')  # 210
sys.path.append('/disk1/hjc/classification/')  # 205

import os
import cv2
from configs import config
from utils import image_util


def mask_dilation(mask, size=10):
    """
    膨胀mask
    @param mask: mask图像
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    mask = cv2.dilate(mask, kernel)
    return mask


def mask_processing():
    for root, dirs, files in os.walk(config.coco_masks):
        print(root)
        for file in files:
            mask_path = os.path.join(root, file)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = mask_dilation(mask, size=32)

            # save
            mask_prd_path = mask_path.replace('masks', 'masks_processed_32')
            save_img(mask_prd_path, mask)


def save_img(img_path, img):
    img_dir = os.path.split(img_path)[0]
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    cv2.imwrite(img_path, img)


def _test():
    img_path = r'D:\Desktop\CV\output\result\vgg16_06221006_images\n02099601\n0209960100001184.jpg'
    mask_path = r'D:\Desktop\CV\output\result\vgg16_06221006_masks\n02099601\n0209960100001184.jpg'
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    # mask = mask_dilation(mask, size=32)
    img = cv2.imread(img_path, cv2.COLOR_BGR2RGB)
    roi = cv2.bitwise_and(img, img, mask=mask)
    image_util.show_cv(roi, 1)
    cv2.imwrite(r'D:\Desktop\test.png', roi)


if __name__ == '__main__':
    _test()
    # mask_processing()
