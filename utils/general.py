import os
from os.path import join

import cv2
import numpy
import numpy as np
import random


def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def issmallobject(bbox, thresh):
    if bbox[0] * bbox[1] <= thresh:
        return True
    else:
        return False


def read_label_txt(label_dir):
    labels = []
    with open(label_dir) as fp:
        for f in fp.readlines():
            labels.append(f.strip().split(' '))
    return labels


def load_txt_label(label_dir):
    return np.loadtxt(label_dir, dtype=str)


def load_txt_labels(label_dir):
    labels = []
    for l in label_dir:
        la = load_txt_label(l)
        labels.append(la)
    return labels


def check_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def drawOneRectAndShow(img, rect):
    """
    @param img: img
    @param rect: ((x_c, y_c), (w, h), theta)
    @return:
    """
    height, width, _ = img.shape
    poly = np.float32(cv2.boxPoints(rect))
    if rect[0][0] < 1 and rect[0][1] < 1 and rect[1][0] < 1 and rect[1][1] < 1:
        poly[:, 0] = poly[:, 0] * width
        poly[:, 1] = poly[:, 1] * height
    poly = np.int0(poly)
    cv2.drawContours(image=img,
                     contours=[poly],
                     contourIdx=-1,
                     color=(255, 0, 0),
                     thickness=2)
    visual(img)


def rescale_yolo_labels(labels, img_shape):
    """
    反归一化 + 转换对角坐标
    @param labels: 标签 cls, x, y, long, short, angle
    @param img_shape:
    @return: [[cls, int(x_left), int(y_left), int(x_right), int(y_right), angle(cv)], ...]
    """
    height, width, _ = img_shape
    rescale_boxes = []
    for box in list(labels):
        cls = box[0]
        rect = \
            longsideformat2cvminAreaRect(float(box[1]), float(box[2]), float(box[3]), float(box[4]), float(box[5])-179.9)
        (x_c, y_c), (w, h), theta = rect

        # print(rect)
        # drawOneRectAndShow(img, rect)

        # # 反归一化
        # x_c = x_c * width
        # y_c = y_c * height
        # w = w * width
        # h = h * height
        # angle = theta
        # 对角坐标
        x1 = x_c - w * .5
        y1 = y_c - h * .5
        x2 = x_c + w * .5
        y2 = y_c + h * .5
        # 反归一化
        x1 *= width
        y1 *= height
        x2 *= width
        y2 *= height
        angle = theta

        # poly = numpy.int0([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
        # rescale_boxes.append([cls, int(x_left), int(y_left), int(x_right), int(y_right), angle])
        rescale_boxes.append([cls, x1, y1, x2, y2, angle])
    return rescale_boxes


def draw_annotation_to_image(img, annotation, save_img_dir):
    for anno in annotation:
        cl, x1, y1, x2, y2 = anno
        cv2.rectangle(img, pt1=(x1, y1), pt2=(x2, y2), color=(255, 0, 0))
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, cl, (int((x1 + x2) / 2), y1 - 5), font, fontScale=0.8, color=(0, 0, 255))
    cv2.imwrite(save_img_dir, img)


def bbox_iou(box1, box2):
    cl, b1_x1, b1_y1, b1_x2, b1_y2 = box1
    if len(box2) == 6:
        cl, b2_x1, b2_y1, b2_x2, b2_y2, _ = box2
    else:
        cl, b2_x1, b2_y1, b2_x2, b2_y2 = box2
    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = max(b1_x1, b2_x1)
    inter_rect_y1 = max(b1_y1, b2_y1)
    inter_rect_x2 = min(b1_x2, b2_x2)
    inter_rect_y2 = min(b1_y2, b2_y2)
    # Intersection area
    inter_width = inter_rect_x2 - inter_rect_x1 + 1
    inter_height = inter_rect_y2 - inter_rect_y1 + 1
    if inter_width > 0 and inter_height > 0:  # strong condition
        inter_area = inter_width * inter_height
        # Union Area
        b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
        b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
        iou = inter_area / (b1_area + b2_area - inter_area)
    else:
        iou = 0
    return iou


def swap(x1, x2):
    if (x1 > x2):
        temp = x1
        x1 = x2
        x2 = temp
    return x1, x2


def norm_sampling(search_space):
    # 随机生成目标中心点
    search_x_left, search_y_left, search_x_right, search_y_right = search_space

    search_x_left = int(search_x_left)
    search_x_right = int(search_x_right)
    search_y_left = int(search_y_left)
    search_y_right = int(search_y_right)

    new_bbox_x_center = random.randint(search_x_left, search_x_right)
    # print(search_y_left, search_y_right, '=')
    new_bbox_y_center = random.randint(search_y_left, search_y_right)
    return [new_bbox_x_center, new_bbox_y_center]


def flip_bbox(roi):
    roi = roi[:, ::-1, :]
    return roi


def sampling_new_bbox_center_point(img_shape):
    height, width, nc = img_shape
    # 修改区域
    search_x_left, search_y_left, search_x_right, search_y_right = \
        width * 0.1, height * 0.1, width * 0.9, height * 0.9
    return [search_x_left, search_y_left, search_x_right, search_y_right]


def random_add_patches(obj_shape, all_boxes, bg_shape, paste_number, iou_thresh, cl_id):
    """
    计算小目标贴到图像上的位置, 并保证bbox不会挡住图片上原有的目标
    @param obj_shape: 小目标shape
    @param all_boxes: 被增强图片上已有目标的对角坐标（包括刚添加的）
    @param bg_shape: 被增强图片shape
    @param paste_number: 将该小目标贴到到原图上的次数, 所以最后添加的总目标数是count * paste_number
    @param iou_thresh: 原图上的bbox和贴上去的roi的bbox的阈值？
    @param cl_id: 小目标的类别下标
    @return: roi在原图上的bbox [cls, x, y, x, y]
    """
    # temp = []
    # for rescale_bbox in rescale_boxes:
    #     temp.append(rescale_bbox)  # 添加被增强图片上已有目标的对角坐标
    obj_h, obj_w, _ = obj_shape
    bg_h, bg_w, _ = bg_shape
    # 防止目标粘贴到图像边缘，缩小小目标可粘贴范围
    center_search_space = sampling_new_bbox_center_point(bg_shape)
    success_num = 0
    new_bboxes = []

    while success_num < paste_number:
        # print(success_num)
        new_bbox_x_center, new_bbox_y_center = norm_sampling(center_search_space)  # 随机生成目标中心点
        # 如果小目标贴到该位置有一半出界，就放弃这个位置
        if new_bbox_x_center - 0.5 * obj_w < 0 or new_bbox_x_center + 0.5 * obj_w > bg_w:
            continue
        if new_bbox_y_center - 0.5 * obj_h < 0 or new_bbox_y_center + 0.5 * obj_h > bg_h:
            continue

        # 小目标对角坐标
        x1, y1, x2, y2 = \
            new_bbox_x_center - 0.5 * obj_w, new_bbox_y_center - 0.5 * obj_h, \
            new_bbox_x_center + 0.5 * obj_w, new_bbox_y_center + 0.5 * obj_h
        new_bbox = [cl_id, int(x1), int(y1), int(x2), int(y2)]

        # TODO 计算IOU
        ious = [bbox_iou(new_bbox, bbox_t) for bbox_t in all_boxes]
        ious2 = [bbox_iou(new_bbox, bbox_t1) for bbox_t1 in new_bboxes]

        if ious2 == []:
            ious2.append(0)

        if max(ious) <= iou_thresh and max(ious2) <= iou_thresh:
            success_num += 1
            # temp.append(new_bbox)
            new_bboxes.append(new_bbox)
        else:
            continue

    return new_bboxes


def update_crops_txt(cropsDir, img_format):
    """
        整理完crops之后，使用该函数更新small.txt文件
    """
    img_list = [os.path.join(cropsDir, f) for f in os.listdir(cropsDir) if f.endswith(img_format)]
    lines = open(join(cropsDir, 'small.txt'), 'r').readlines()
    update = []
    for line in lines:
        fileDir, cls = line.strip().split(' ')
        if fileDir in img_list:
            update.append(fileDir + ' ' + cls)
    update_file = open(join(cropsDir, 'small.txt'), 'w')
    for item in update:
        print(item)
        update_file.writelines(item + '\n')
    update_file.close()
    print("Update Success ✅")


def rotate_bound(image, angle):
    """
    根据angle旋转图像
    """
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))


def visual(img):
    cv2.imshow("", img)
    cv2.waitKey(0)


def longsideformat2cvminAreaRect(x_c, y_c, longside, shortside, theta_longside):
    """
    trans longside format(x_c, y_c, longside, shortside, θ) to minAreaRect(x_c, y_c, width, height, θ)
    两者区别为:
            当opencv表示法中width为最长边时（包括正方形的情况），则两种表示方法一致
            当opencv表示法中width不为最长边 ，则最长边表示法的角度要在opencv的Θ基础上-90度
    @param x_c: center_x
    @param y_c: center_y
    @param longside: 最长边
    @param shortside: 最短边
    @param theta_longside: 最长边和x轴逆时针旋转的夹角，逆时针方向角度为负 [-180, 0)
    @return: ((x_c, y_c),(width, height),Θ)
            x_c: center_x
            y_c: center_y
            width: x轴逆时针旋转碰到的第一条边最长边
            height: 与width不同的边
            theta: x轴逆时针旋转与width的夹角，由于原点位于图像的左上角，逆时针旋转角度为负 [-90, 0)
    """
    if (theta_longside >= -180 and theta_longside < -90):  # width is not the longest side
        width = shortside
        height = longside
        theta = theta_longside + 90
    else:
        width = longside
        height = shortside
        theta = theta_longside

    if theta < -90 or theta >= 0:
        print('当前θ=%.1f，超出opencv的θ定义范围[-90, 0)' % theta)

    return (x_c, y_c), (width, height), theta


def cvminAreaRect2longsideformat(x_c, y_c, width, height, theta):
    '''
    trans minAreaRect(x_c, y_c, width, height, θ) to longside format(x_c, y_c, longside, shortside, θ)
    两者区别为:
            当opencv表示法中width为最长边时（包括正方形的情况），则两种表示方法一致
            当opencv表示法中width不为最长边 ，则最长边表示法的角度要在opencv的Θ基础上-90度
    @param x_c: center_x
    @param y_c: center_y
    @param width: x轴逆时针旋转碰到的第一条边
    @param height: 与width不同的边
    @param theta: x轴逆时针旋转与width的夹角，由于原点位于图像的左上角，逆时针旋转角度为负 [-90, 0)
    @return:
            x_c: center_x
            y_c: center_y
            longside: 最长边
            shortside: 最短边
            theta_longside: 最长边和x轴逆时针旋转的夹角，逆时针方向角度为负 [-180, 0)
    '''
    '''
    意外情况:(此时要将它们恢复符合规则的opencv形式：wh交换，Θ置为-90)
    竖直box：box_width < box_height  θ=0
    水平box：box_width > box_height  θ=0
    '''
    if theta == 0:
        theta = -90
        buffer_width = width
        width = height
        height = buffer_width

    if theta > 0:
        if theta != 90:  # Θ=90说明wh中有为0的元素，即gt信息不完整，无需提示异常，直接删除
            print('θ计算出现异常，当前数据为：%.16f, %.16f, %.16f, %.16f, %.1f;超出opencv表示法的范围：[-90,0)' % (x_c, y_c, width, height, theta))
        return False

    if theta < -90:
        print('θ计算出现异常，当前数据为：%.16f, %.16f, %.16f, %.16f, %.1f;超出opencv表示法的范围：[-90,0)' % (x_c, y_c, width, height, theta))
        return False

    if width != max(width, height):  # 若width不是最长边
        longside = height
        shortside = width
        theta_longside = theta - 90
    else:  # 若width是最长边(包括正方形的情况)
        longside = width
        shortside = height
        theta_longside = theta

    if longside < shortside:
        print('旋转框转换表示形式后出现问题：最长边小于短边;[%.16f, %.16f, %.16f, %.16f, %.1f]' % (x_c, y_c, longside, shortside, theta_longside))
        return False
    if (theta_longside < -180 or theta_longside >= 0):
        print('旋转框转换表示形式时出现问题:θ超出长边表示法的范围：[-180,0);[%.16f, %.16f, %.16f, %.16f, %.1f]' % (x_c, y_c, longside, shortside, theta_longside))
        return False

    return x_c, y_c, longside, shortside, theta_longside

# if __name__ == "__main__":
# update_crops_txt('./data/crops', 'png')
