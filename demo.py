import sys

sys.path.append('utils')
import os
import random
from os.path import join
from utils.aug import *
from utils.general import check_dir

# ###########Pipeline##############
"""
1 准备数据集和yolo格式标签, 如果自己的数据集是voc或coco格式的，先转换成yolo格式，增强后在转回来
2 run crop_image.py  裁剪出目标并保存图片
3 run demo.py   随机将裁剪出目标图片贴到需要增强的数据集上，并且保存增强后的图片集和label文件
"""


def copy_paste():
    class2id = {'plane': 0, 'baseball-diamond': 1, 'bridge': 2, 'ground-track-field': 3, 'small-vehicle': 4,
                'large-vehicle': 5, 'ship': 6, 'tennis-court': 7, 'basketball-court': 8, 'storage-tank': 9,
                'soccer-ball-field': 10, 'roundabout': 11, 'harbor': 12,
                'swimming-pool': 13, 'helicopter': 14, 'container-crane': 15}

    cl = 'small-vehicle'  # 在这里更改 你要转换的类别
    times = 15  # 更改每次在 原图上 添加多少个小图像

    base_dir = 'augmentation'

    cl_id = class2id[cl]
    print('cl_id', cl_id)

    save_pic = join(base_dir, 'images')
    save_txt = join(base_dir, 'txt')

    check_dir(save_pic)
    check_dir(save_txt)

    # 获取图像的路径，以及图像对应框框的标签
    img_dir = [os.path.join('background', f) for f in os.listdir('background') if f.endswith('png')]
    labels_dir = [os.path.join('background', f) for f in os.listdir('background') if f.endswith('txt')]

    small_label_dir = [f.strip() for f in open('crops/small.txt').readlines()]
    random.shuffle(small_label_dir)

    for image_dir, label_dir in zip(img_dir, labels_dir):
        print(image_dir, label_dir)
        small_img = []
        for x in range(times):
            if not small_label_dir:
                small_label_dir = [f.strip() for f in open(join(base_dir, 'crops/small.txt')).readlines()]
                random.shuffle(small_label_dir)
            small_img.append(small_label_dir.pop())
        # print("ok")
        copysmallobjects(image_dir, label_dir, save_pic, save_txt, small_img, times, cl_id)


if __name__ == "__main__":
    copy_paste()
