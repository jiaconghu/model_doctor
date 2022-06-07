import sys

# sys.path.append('/disk2/hjc/classification/')  # 210
sys.path.append('/disk1/hjc/classification/')  # 205

from pycocotools.coco import COCO
import os
import cv2
import numpy as np
import skimage.measure

from utils import image_util
from configs import config

# dataDir = config.datasets_coco
dataDir = r'/datasets/COCO2017'
dataType = 'train'
annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType + '2017')
imgFile = '{}/images/{}'.format(dataDir, dataType)
coco = COCO(annFile)


def count_cats():
    cats = coco.loadCats(coco.getCatIds())
    nms = [cat['name'] for cat in cats]
    nms_num = []

    for nm in nms:
        cat_ids = coco.getCatIds(catNms=[nm])
        img_ids = coco.getImgIds(catIds=cat_ids)
        nms_num.append((nm, len(img_ids)))

    nms_num.sort(key=lambda nm_len: nm_len[1], reverse=True)
    print(nms_num)
    return nms_num


def gen_mask(nms):
    for nm in nms:
        print('=' * 40)
        print('cat_name:' + nm)
        cat_ids = coco.getCatIds(catNms=[nm])
        print(cat_ids)
        img_ids = coco.getImgIds(catIds=cat_ids)
        print(len(img_ids))
        # print('-' * 40)

        cnt = 0
        for i in range(len(img_ids)):
            img = coco.loadImgs(img_ids[i])[0]
            img_name = os.path.splitext(img["file_name"])[0]

            ann_ids = coco.getAnnIds(imgIds=img['id'], catIds=cat_ids, iscrowd=None)
            anns = coco.loadAnns(ann_ids)

            mask = np.zeros(shape=(img['height'], img['width']), dtype=np.uint8)
            for ann in anns:
                mask += coco.annToMask(ann)
            bbox, size = measure_bbox(mask, padding=10)

            image = cv2.imread(os.path.join(imgFile, img["file_name"]))
            if size > 224 * 224:
                cnt += 1
                mask_path = os.path.join(config.coco_masks, dataType, nm, "{}.png".format(img_name))
                # image_util.show_cv(mask * 255)
                # mask = mask[bbox[0]:bbox[2], bbox[1]:bbox[3]]
                # image_util.show_cv(mask * 255)
                save_img(mask_path, mask * 255)

                image_path = os.path.join(config.coco_images, dataType, nm, "{}.png".format(img_name))
                # image_util.show_cv(image_ori)
                # image = image_ori[bbox[0]:bbox[2], bbox[1]:bbox[3]]
                # image_util.show_cv(image)
                save_img(image_path, image)
        print('count:', cnt)


def measure_bbox(mask, padding=0):
    mask_label, num = skimage.measure.label(
        mask, connectivity=1, background=0, return_num=True)
    mask_region = skimage.measure.regionprops(mask_label)
    max_region_area = 0
    max_region_idx = 0
    for i in range(num):
        if mask_region[i].area > max_region_area:
            max_region_area = mask_region[i].area
            max_region_idx = i
    bbox = mask_region[max_region_idx].bbox  # (min_y,min_x,max_y,max_x)
    bbox_area = mask_region[max_region_idx].bbox_area  # ((max_y-min_y)*(max_x-min_x))
    # print(bbox)
    # print(area)
    # print(mask.shape)

    bbox_padding = [0, 0, 0, 0]
    bbox_padding[0] = bbox[0] - padding if bbox[0] - padding > 0 else 0
    bbox_padding[1] = bbox[1] - padding if bbox[1] - padding > 0 else 0
    bbox_padding[2] = bbox[2] + padding if bbox[2] + padding < mask.shape[0] else mask.shape[0]
    bbox_padding[3] = bbox[3] + padding if bbox[3] + padding < mask.shape[1] else mask.shape[1]
    return bbox_padding, bbox_area


def save_img(img_path, img):
    img_dir = os.path.split(img_path)[0]
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    cv2.imwrite(img_path, img)


if __name__ == '__main__':
    nms_num = count_cats()
    # select top50 nms
    # nms_topn = []
    # for nm in nms_num[1:]:
    #     nms_topn.append(nm[0])

    # nms_topn = ['airplane', 'banana', 'bear', 'bed', 'bird', 'boat', 'broccoli', 'bus', 'cake', 'cat', 'cow',
    #             'dog', 'donut', 'elephant', 'fire hydrant', 'giraffe', 'horse', 'hot dog', 'motorcycle', 'pizza',
    #             'refrigerator', 'scissors', 'sheep', 'teddy bear', 'toilet', 'train', 'tv', 'zebra']

    nms_topn = ['airplane', 'bus', 'cat', 'dog', 'elephant', 'giraffe', 'horse', 'motorcycle', 'pizza', 'teddy bear',
                'train', 'zebra']
    print(nms_topn)
    gen_mask(nms_topn)
