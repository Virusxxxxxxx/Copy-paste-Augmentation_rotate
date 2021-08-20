import glob
from utils.general import *
import os
import sys


def clean_crops():
    img_dir = [os.path.join('.\\crops', f) for f in os.listdir('crops') if f.endswith('png')]
    f = open('./crop/small.txt', 'w')
    for path in img_dir:
        print(path)

        f.write(path + '\n')
    f.close()


def find_min_size(img_list):
    h_min = 2000  # 最短高
    w_min = 2000  # 最短宽
    for item in img_list:
        img = cv2.imread(item)
        h, w, _ = img.shape
        if h < h_min:
            h_min = h
        if w < w_min:
            w_min = w

    return h_min, w_min


def crop_img(data_root, txt_save, save_dir='./data/crops', img_format='png'):
    """
    根据bbox截取目标roi，并保存图片，生成截取后的小图片
    :param data_root: 存放原始图片
    :param txt_save: 存放yolo格式label
    :param save_dir: 截取下来的小图片默认存放在 ./crops 文件夹中
    :param img_format: 图片后缀
    :return:
    """

    check_dir(save_dir)

    img_list = glob.glob(data_root + "/*." + img_format)
    txt_list = glob.glob(txt_save + "/*.txt")

    crops_txt = open('data/crops/small.txt', "w", encoding='utf-8')  # 截图下来的小图片存放的路径

    # max_s = -1
    # min_s = 1000

    h_min, w_min = find_min_size(img_list)  # 找到被切割图片最小宽高
    print('h_min', h_min, 'w_min', w_min)

    count = 0
    for img_path, txt_path in zip(img_list, txt_list):

        img_name = os.path.basename(img_path)  # ***.png
        img_label = open(txt_path, "r")
        img = cv2.imread(img_path)

        # 防止图片为空的情况，因为这样图片打不开
        if img is not None:
            height, width, channel = img.shape

            labels = img_label.readlines()

            for num, label in enumerate(labels):
                # print(num, file_content)
                clss, xc, yc, w, h = label.split()
                xc, yc, w, h = float(xc), float(yc), float(w), float(h)
                # 反归一化
                xc *= width
                yc *= height
                w *= width
                h *= height
                # 保存最大/小面积
                # max_s = max(w * h, max_s)
                # min_s = min(w * h, min_s)
                # 对角坐标
                x1, y1 = int(xc - w / 2), int(yc - h / 2)
                x2, y2 = int(xc + w / 2), int(yc + h / 2)

                crop = img[y1:y2, x1:x2]  # 裁剪小图标

                # h, w, _ = crop_img.shape
                # if h > h_min or w > w_min:
                #     crop_img = cv2.resize(crop_img,(math.ceil(h*0.5),math.ceil(w*0.5)))

                crop_name = img_name.split('.')[0] + "_crop_" + str(num) + "." + img_format

                print('{} image size is {}'.format(crop_name, crop.shape))

                cv2.imwrite(os.path.join(save_dir, crop_name), crop)
                # cv2.imshow("croped",crop_img)
                # cv2.waitKey(0)

                # 文件名写入txt
                crops_txt.write(os.path.join(save_dir, crop_name) + "\n")
                count += 1
        img_label.close()
    crops_txt.close()
    print('Total number of pictures:', len(img_list))
    print("Total crop:", count)
    # print(max_s, min_s)

    # 返回保存截图后小图标的 文件夹路径
    # return save_dir


if __name__ == '__main__':
    data_root = './data/images'
    txt_save = './data/yolo_labels'
    crop_img(data_root, txt_save, img_format='png')
