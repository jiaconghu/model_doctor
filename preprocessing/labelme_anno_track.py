import os
from configs import config
from utils import file_util


def track(track_path, result_path):
    for root, _, files in os.walk(track_path):
        for file in files:
            if os.path.splitext(file)[1] == '.json':
                class_name = os.path.split(root)[-1]
                src = os.path.join(root, file)
                dst = os.path.join(result_path, class_name, file)
                file_util.copy_file(src, dst)


def main():
    track_path = os.path.join(config.output_result, 'vgg16_08241356', 'images')
    result_path = os.path.join(config.output_result, 'mnim_lc_images')
    track(track_path, result_path)


if __name__ == '__main__':
    main()
