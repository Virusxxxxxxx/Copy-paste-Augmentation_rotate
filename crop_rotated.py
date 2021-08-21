# -*- coding:utf-8 -*-
"""旋转图像并剪裁"""
import glob
from os.path import join
import cv2
from math import *
import numpy as np
from utils.general import *


def crop_img_rotated(imgDir, labelDir, save_dir='./data/crops', img_format='png'):
    check_dir(save_dir)
    crops_txt = open('data/crops/small.txt', "w", encoding='utf-8')  # 截图下来的小图片存放的路径
    img_list = glob.glob(imgDir + "/*." + img_format)
    txt_list = glob.glob(labelDir + "/*.txt")
    print('Total number of pictures:', len(img_list))

    count = 0
    for img_path, txt_path in zip(img_list, txt_list):
        img_name = os.path.basename(img_path)  # ***.png
        img_label = open(txt_path, "r")
        img = cv2.imread(img_path)

        # 防止图片为空的情况，因为这样图片打不开
        if img is not None:
            labels = img_label.readlines()

            for num, label in enumerate(labels):
                items = label.strip().split(' ')
                cls = items[0]
                (x_c, y_c), (width, height), theta = \
                    longsideformat2cvminAreaRect(float(items[1]), float(items[2]), float(items[3]),
                                                 float(items[4]), float(items[5]) - 179.9)
                crop_img = rotate(img, x_c, y_c, width, height, theta)
                # cv2.imshow("", crop_img)
                # cv2.waitKey(0)

                # 裁剪后的图片名
                crop_name = img_name.split('.')[0] + "_crop_" + str(num) + "." + img_format
                cv2.imwrite(join(save_dir, crop_name), crop_img)  # 裁减得到的旋转矩形框
                crops_txt.write(os.path.join(save_dir, crop_name) + " " + cls + "\n")  # 文件名写入txt
                count += 1
        img_label.close()
    crops_txt.close()
    print('Total number of pictures:', len(img_list))
    print("Total crop:", count)


def rotate(img, x_c, y_c, w, h, angle):
    # check_dir(cropDir)

    print(x_c, y_c, w, h, angle)

    height = img.shape[0]  # 原始图像高度
    width = img.shape[1]  # 原始图像宽度
    rotateMat = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)  # 按angle角度旋转图像
    heightNew = int(width * fabs(sin(radians(angle))) + height * fabs(cos(radians(angle))))
    widthNew = int(height * fabs(sin(radians(angle))) + width * fabs(cos(radians(angle))))

    rotateMat[0, 2] += (widthNew - width) / 2
    rotateMat[1, 2] += (heightNew - height) / 2
    imgRotation = cv2.warpAffine(img, rotateMat, (widthNew, heightNew), borderValue=(255, 255, 255))
    # 反归一化
    x_c *= width
    w *= width
    y_c *= height
    h *= height
    # 四角坐标
    poly = np.float32(cv2.boxPoints(((x_c, y_c), (w, h), angle)))
    # poly = [(x1,y1),(x2,y2),(x3,y3),(x4,y4)]

    pt1 = poly[0]
    pt2 = poly[1]
    pt3 = poly[2]
    pt4 = poly[3]

    # 旋转后图像的四点坐标
    [[pt1[0]], [pt1[1]]] = np.dot(rotateMat, np.array([[pt1[0]], [pt1[1]], [1]]))
    [[pt3[0]], [pt3[1]]] = np.dot(rotateMat, np.array([[pt3[0]], [pt3[1]], [1]]))
    [[pt2[0]], [pt2[1]]] = np.dot(rotateMat, np.array([[pt2[0]], [pt2[1]], [1]]))
    [[pt4[0]], [pt4[1]]] = np.dot(rotateMat, np.array([[pt4[0]], [pt4[1]], [1]]))

    # 处理反转的情况
    if pt2[1] > pt4[1]:
        pt2[1], pt4[1] = pt4[1], pt2[1]
    if pt1[0] > pt3[0]:
        pt1[0], pt3[0] = pt3[0], pt1[0]

    imgOut = imgRotation[int(pt2[1]):int(pt4[1]), int(pt1[0]):int(pt3[0])]
    # cv2.imshow("", imgOut)
    # cv2.waitKey(0)

    return imgOut  # rotated image


# 　根据四点画原矩形
def drawRect(img, pt1, pt2, pt3, pt4, color, lineWidth):
    cv2.line(img, pt1, pt2, color, lineWidth)
    cv2.line(img, pt2, pt3, color, lineWidth)
    cv2.line(img, pt3, pt4, color, lineWidth)
    cv2.line(img, pt1, pt4, color, lineWidth)


if __name__ == "__main__":
    data_root = './data/images'
    txt_save = './data/yolo_labels_rotated'
    crop_img_rotated(data_root, txt_save, img_format='png')
