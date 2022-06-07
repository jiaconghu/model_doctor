import sys

sys.path.append('/workspace/classification/code/')  # zjl
import csv
import os
from PIL import Image

from configs import config

data_dir = '/workspace/classification/datasets/mini-imagenet/'
train_csv_path = data_dir + '/train.csv'
val_csv_path = data_dir + '/val.csv'
test_csv_path = data_dir + '/test.csv'
images_path = data_dir + '/images'

output_images = config.data_mini_imagenet

train_label = {}
val_label = {}
test_label = {}
with open(train_csv_path) as csv_file:
    csv_reader = csv.reader(csv_file)
    birth_header = next(csv_reader)
    for row in csv_reader:
        train_label[row[0]] = row[1]

with open(val_csv_path) as csv_file:
    csv_reader = csv.reader(csv_file)
    birth_header = next(csv_reader)
    for row in csv_reader:
        val_label[row[0]] = row[1]

with open(test_csv_path) as csv_file:
    csv_reader = csv.reader(csv_file)
    birth_header = next(csv_reader)
    for row in csv_reader:
        test_label[row[0]] = row[1]

for filename in os.listdir(images_path):
    if not filename.endswith('jpg'):
        continue

    path = images_path + '/' + filename
    print(path)
    im = Image.open(path)

    if filename in train_label.keys():
        tmp = train_label[filename]
        temp_path = output_images + '/train' + '/' + tmp
        if not os.path.exists(temp_path):
            os.makedirs(temp_path)
        t = temp_path + '/' + filename
        im.save(t)

    elif filename in val_label.keys():
        tmp = val_label[filename]
        temp_path = output_images + '/val' + '/' + tmp
        if not os.path.exists(temp_path):
            os.makedirs(temp_path)
        t = temp_path + '/' + filename
        im.save(t)

    elif filename in test_label.keys():
        tmp = test_label[filename]
        temp_path = output_images + '/test' + '/' + tmp
        if not os.path.exists(temp_path):
            os.makedirs(temp_path)
        t = temp_path + '/' + filename
        im.save(t)
