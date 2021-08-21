import cv2 as cv2
import numpy as np
import random
from os.path import basename, join, dirname
from general import *


def find_str(filename):
    if 'train' in filename:
        return dirname(filename[filename.find('train'):])
    else:
        return dirname(filename[filename.find('val'):])


def doublePoly2longsideformat(img_shape, box):
    height, width, _ = img_shape
    cls, x1, y1, x2, y2, angle = box

    h = y2 - y1
    w = x2 - x1
    x_c = x1 + w * .5
    y_c = y1 + h * .5

    rect = ((x_c, y_c), (w, h), angle)
    # drawOneRectAndShow(img, rect)

    # 归一化
    x_c /= width
    y_c /= height
    h /= height
    w /= width

    x_c, y_c, longside, shortside, angle = cvminAreaRect2longsideformat(x_c, y_c, w, h, angle)
    theta_label = int(angle + 180.5)  # range int[0,180] 四舍五入
    if theta_label == 180:  # range int[0,179]
        theta_label = 179
    return cls, x_c, y_c, longside, shortside, theta_label


def convert_all_boxes(img_shape, labels, output_label_dir):
    label_file = open(output_label_dir, 'w')
    for label in labels:
        cls, x1, y1, x2, y2, angle = label
        box = (cls, float(x1), float(y1), float(x2), float(y2), float(angle))
        n_box = doublePoly2longsideformat(img_shape, box)
        # bb = convert((width, height), b)
        label_file.write(" ".join([str(a) for a in n_box]) + '\n')
    label_file.close()


def save_crop_image(save_crop_base_dir, image_dir, idx, roi):
    crop_save_dir = join(save_crop_base_dir, find_str(image_dir))
    check_dir(crop_save_dir)
    crop_img_save_dir = join(
        crop_save_dir,
        basename(image_dir)[:-3] + '_crop_' + str(idx) + '.jpg')
    cv2.imwrite(crop_img_save_dir, roi)


def GaussianBlurImg(image):
    # 高斯模糊
    ran = random.randint(0, 9)
    if ran % 2 == 1:
        image = cv2.GaussianBlur(image, ksize=(ran, ran), sigmaX=0, sigmaY=0)
    else:
        pass
    return image


def roi_resize(image, small_pic_width, small_pic_height):
    # 改变小图片大小
    image = cv2.resize(image, (small_pic_width, small_pic_height),
                       interpolation=cv2.INTER_AREA)  # 注意，目标size不能太大，否则图片会不够大小贴下目标
    return image


def get_resolution(bg_shape, obj_shape):
    """
    获得背景图的分辨率以调整 粘贴小图像的大小，防止因小图像太大导致粘贴不上去
    @param bg_shape: 当前被增强图片的shape
    @param obj_shape: 当前被增强图片的shape
    @return: obj_h, obj_w
    """

    h1, w1, _ = bg_shape
    h2, w2, _ = obj_shape
    # print('height', height, 'width', width)
    obj_max_size = max(h2, w2)
    bg_min_size = min(h1, w1)

    if obj_max_size > bg_min_size * 0.5:  # 如果obj的最长边比bg的最短边的一半大
        scale = bg_min_size * 0.25 / obj_max_size  # 缩放系数
        return h2 * scale, w2 * scale
    else:
        return h2, w2


def copysmallobjects(image_dir, label_dir, save_pic, save_txt, small_img_dir, count):
    """
    copy-paste增强
    @param image_dir: 当前被增强图片dir './data/images\\P0706__1__0___0.png'
    @param label_dir: 当前被增强图片标签dir './data/yolo_labels_rotated\\P0706__1__0___0.txt'
    @param save_pic: 增强后图片dir './output/images'
    @param save_txt: 增强后图片标签dir './output/labels'
    @param small_img_dir: 要添加的小目标dir cls
    @param count: 添加个数
    @return:
    """
    check_dir(save_txt)
    check_dir(save_pic)

    image = cv2.imread(image_dir)
    image_name = image_dir.split('\\')[-1].split('.')[0]
    print('当前被增强图片:', image_name)

    labels = read_label_txt(label_dir)  # 读取标签
    if len(labels) == 0:
        return

    bg_height, bg_width, _ = image.shape
    # 分辨率太低的图片会被过滤掉
    if bg_height <= 200 and bg_width <= 200:
        return

    # 反归一化 + 转化为对角坐标, angle(cv)
    rescale_labels = rescale_yolo_labels(labels, image.shape)
    # print("org bbox:", rescale_labels)  # 原图像bbox集合

    all_boxes = []  # 所有目标的box（包括新加入的小目标）
    for rescale_label in rescale_labels:
        all_boxes.append(rescale_label)

    for item in small_img_dir:
        small_obj_dir, small_obj_cls = item.split(' ')
        small_obj_img = cv2.imread(small_obj_dir)

        # 根据背景图调整小目标的大小
        small_obj_h, small_obj_w = get_resolution(image.shape, small_obj_img.shape)
        # 调整小目标的大小
        roi = cv2.resize(small_obj_img, (int(small_obj_w), int(small_obj_h)), interpolation=cv2.INTER_AREA)

        # 得到小目标贴到图像上的位置, 并保证bbox不会挡住图片上原有的目标
        new_bboxes = random_add_patches(roi.shape, all_boxes, image.shape,
                                        paste_number=1, iou_thresh=0, cl_id=small_obj_cls)
        # new bboxes = [[cls, x1, y1, x2, y2], [cls, x1, y1, x2, y2], ...]

        # print('{} new_bbox'.format(image_dir), new_bboxes)

        # 开始绘制
        count = 0
        for new_bbox in new_bboxes:
            count += 1

            cl, x1, y1, x2, y2 = new_bbox[0], new_bbox[1], new_bbox[2], new_bbox[3], new_bbox[4]
            x_c, y_c = int((x2 + x1) / 2), int((y2 + y1) / 2)  # obj中心点坐标

            try:
                # 随机旋转角度
                angle = -random.randint(0, 90)
                rotate_roi = rotate_bound(roi, angle)  # 旋转后的roi
                # visual(rotate_roi)

                # 旋转之后，图像大小发生变化，背景作图区域就需要跟着变化，但是标签不变
                # roi = GaussianBlurImg(roi)  # 高斯模糊
                center = x_c, y_c
                # mask = 255 * np.ones(rotate_roi.shape, rotate_roi.dtype)  # 创建一个全白mask
                # mask = np.zeros(rotate_roi.shape, rotate_roi.dtype)  # 创建一个全白mask
                mask = (np.ceil(rotate_roi / 255.0) * 255.0).astype('uint8')  # 提取出小目标的区域作为mask

                image = cv2.seamlessClone(rotate_roi, image, mask, center, cv2.NORMAL_CLONE)
                new_bbox.append(angle)
                all_boxes.append(new_bbox)
                # visual(image)

                # print("end try")
            except cv2.Error:
                print("Error: cv2.error happend in {}".format(small_obj_dir.split('\\')[-1]))
                continue

    # print('before {} is ok'.format(image_dir))
    # dir_name = find_str(image_dir)
    # yolo_txt_dir = join(save_txt, basename(image_dir.replace('.png', '_aug_%s.txt' % str(count))))
    # cv2.imwrite(join(save_pic, basename(image_dir).replace('.png', '_aug_%s.png' % str(count))), image)

    output_label_dir = join(save_txt, image_name + '.txt')  # 完整标签路径
    output_img_dir = join(save_pic, image_dir.split('\\')[-1])  # 完整图片路径
    # visual(image)
    cv2.imwrite(output_img_dir, image)

    convert_all_boxes(image.shape, all_boxes, output_label_dir)

