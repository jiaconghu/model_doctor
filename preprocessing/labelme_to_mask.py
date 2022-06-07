import os
import json
import numpy as np
import cv2

from configs import config
from utils import image_util


def parse_json(json_path):
    data = json.load(open(json_path))
    shapes = data['shapes'][0]
    points = shapes['points']
    return points


def polygons_to_mask(img_shape, polygons):
    """
    边界点生成mask
    :param img_shape: [h,w]
    :param polygons: labelme JSON中的边界点格式 [[x1,y1],[x2,y2],[x3,y3],...[xn,yn]]
    :return: mask 0-1
    """
    mask = np.zeros(img_shape, dtype=np.uint8)
    polygons = np.asarray([polygons], np.int32)  # 这里必须是int32，其他类型使用fillPoly会报错
    cv2.fillPoly(mask, polygons, 1)  # 非int32 会报错
    return mask


def main():
    images_dir = os.path.join(config.output_result, 'vgg16_09012200', 'images')
    for root, _, files in os.walk(images_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.json':
                img = cv2.imread(os.path.join(root, file.replace('json', 'png')))

                json_name = os.path.splitext(file)[0]
                json_path = os.path.join(root, file)
                print(json_path)
                mask = polygons_to_mask([img.shape[0], img.shape[1]], parse_json(json_path)) * 255
                mask_path = os.path.join(config.result_masks_stl10, json_name + '.png')
                image_util.save_cv(mask, mask_path)


if __name__ == '__main__':
    main()
