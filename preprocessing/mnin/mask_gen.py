import sys

sys.path.append('/disk2/hjc/classification/')  # 210
import os
import numpy as np
import cv2

from configs import config
from utils import image_util


def iter_img_to_gray():
    for root, _, files in os.walk(config.result_images):
        for file in files:
            path = os.path.join(root, file)
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            img = rgb_to_gray(img)
            filename = path.replace(config.result_images, config.result_images + '_gray')
            image_util.save_cv(img, filename)


def iter_img_to_mask():
    for root, _, files in os.walk(config.result_images + '_gray'):
        for file in files:
            path = os.path.join(root, file)
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            img = gen_mask(img)
            filename = path.replace(config.result_images + '_gray', config.result_masks)
            image_util.save_cv(img, filename)


def rgb_to_gray(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray


def gen_mask(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_red = np.array([156, 43, 46])
    upper_red = np.array([180, 255, 255])
    mask_red = cv2.inRange(img_hsv, lower_red, upper_red)

    lower_blue = np.array([78, 43, 46])
    upper_blue = np.array([99, 255, 255])
    mask_blue = cv2.inRange(img_hsv, lower_blue, upper_blue)

    lower_yellow = np.array([26, 43, 46])
    upper_yellow = np.array([34, 255, 255])
    mask_yellow = cv2.inRange(img_hsv, lower_yellow, upper_yellow)

    outline = mask_red | mask_blue | mask_yellow

    fill = outline.copy()
    h, w = img.shape[:2]
    zeros = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(fill, zeros, (0, 0), 255)  # flood fill

    fill = cv2.bitwise_not(fill)
    mask = outline | fill
    return mask


if __name__ == '__main__':
    iter_img_to_gray()
    # iter_img_to_mask()
