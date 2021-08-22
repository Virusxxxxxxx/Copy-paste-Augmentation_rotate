import argparse
import sys
sys.path.append('utils')
from utils.YOLO_Transform import drawLongsideFormatimg
import utils.dota_utils as util
from utils.aug import *


# ###########Pipeline##############
"""
1 准备数据集和yolo旋转标签，放在./data
2 run crop_rotated.py  裁剪出目标并保存图片
3 run copy_paste.py   随机将裁剪出目标图片贴到需要增强的数据集上，并且保存增强后的图片集和label文件
"""
# class2id = {'plane': 0, 'baseball-diamond': 1, 'bridge': 2, 'ground-track-field': 3, 'small-vehicle': 4,
#             'large-vehicle': 5, 'ship': 6, 'tennis-court': 7, 'basketball-court': 8, 'storage-tank': 9,
#             'soccer-ball-field': 10, 'roundabout': 11, 'harbor': 12,
#             'swimming-pool': 13, 'helicopter': 14, 'container-crane': 15}


def copy_paste(opt):
    imgDir = opt.imgDir  # 待增强的图片路径
    labelDir = opt.labelDir  # 待增强的图片标签路径
    outputDir = opt.outputDir
    img_format = opt.img_format
    crops_dir = opt.crops_dir
    count = int(opt.count)  # 更改每次在原图上添加多少个小目标

    # 指定增强后的输出路径
    save_pic = join(outputDir, 'images')
    save_txt = join(outputDir, 'labels')
    check_dir(save_pic)
    check_dir(save_txt)

    # 获取待增强的图像路径，以及标签路径
    imgs_dir = [os.path.join(imgDir, f) for f in os.listdir(imgDir) if f.endswith(img_format)]
    labels_dir = [os.path.join(labelDir, f) for f in os.listdir(labelDir) if f.endswith('txt')]

    small_dir = join(crops_dir, 'small.txt')
    small = [f.strip() for f in open(small_dir).readlines()]
    random.shuffle(small)  # 打乱顺序

    for image, label in zip(imgs_dir, labels_dir):
        # print(image, label)
        small_img = []  # 要添加的小目标列表
        for x in range(count):  # 从小目标中随机取count个目标加入图片
            # 每次从small里取一个并出栈，一轮取完之后如果还不够count，就重新再读一次文件
            if small == []:
                small = [f.strip() for f in open(small_dir).readlines()]
                random.shuffle(small)
            small_img.append(small.pop())
        copysmallobjects(image, label, save_pic, save_txt, small_img, count)


if __name__ == "__main__":
    opt = parse_opt()
    # 更新筛选后的裁剪图片
    update_crops_txt(opt.crops_dir, opt.img_format)

    copy_paste(opt)

    # 画图验证标签正确性
    drawLongsideFormatimg(outputPath=opt.outputDir,
                          extractclassname=util.classnames_v1_5)
