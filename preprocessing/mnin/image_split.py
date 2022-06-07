import sys

sys.path.append('/workspace/classification/code/')  # zjl
# sys.path.append('/nfs3-p1/hjc/classification/code/')  # vipa

import os
import random

from configs import config
from utils import file_util


def split():
    SplitData(input_path=config.data_mini_imagenet_temp,
              output_path=config.data_mini_imagenet)


class SplitData:
    def __init__(self, input_path, output_path, percentage=0.2):
        self.input_path = input_path
        self.output_path = output_path
        self.percentage = percentage  # this ratio represents the proportion of the test set
        self._split_data(self._data_idx())

    def _data_idx(self):
        data_idx = {}

        for root, _, files in sorted(os.walk(self.input_path)):
            if len(files) != 0:
                class_name = os.path.split(root)[1]
                data_idx[class_name] = self._random_idx(start=0, end=len(files))
                # if len(data_random_idx) == 10:  # select 10 classes
                #     break
        return data_idx

    def _split_data(self, data_random_idx):
        for root, _, files in sorted(os.walk(self.input_path)):
            data_idx = 0
            for file in sorted(files):
                class_name = os.path.split(root)[1]
                if class_name in data_random_idx.keys():
                    if data_idx in data_random_idx[class_name]:
                        data_type = 'test'
                    else:
                        data_type = 'train'

                    original_path = os.path.join(root, file)
                    output_path = os.path.join(self.output_path, data_type, class_name, file)
                    file_util.copy_file(original_path, output_path)

                    data_idx += 1
                    print(root, data_idx, data_type)

    def _random_idx(self, start, end):
        random.seed(0)
        start, end = (int(start), int(end)) if start <= end else (int(end), int(start))
        length = int((end - start) * self.percentage)
        ran_list = []
        while len(ran_list) < length:
            x = random.randint(start, end)
            if x not in ran_list:
                ran_list.append(x)
        return ran_list


if __name__ == '__main__':
    split()
