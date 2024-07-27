import argparse
import glob
import json
import os
import random
import shutil
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch
from PIL import Image
from prettytable import PrettyTable
from sklearn.metrics import roc_curve, roc_auc_score, auc, f1_score
from tqdm import tqdm
from sklearn.metrics import confusion_matrix


def get_path(path_int):
    '''
    :param path_int: 0表示获取当前路径，1表示当前路径的上一次路径，2表示当前路径的上2次路径，以此类推
    :return: 返回我们需要的绝对路径，是双斜号的绝对路径
    '''
    path_count=path_int
    path_current=os.path.abspath(r".")
    # print('path_current=',path_current)
    path_current_split=path_current.split('\\')
    # print('path_current_split=',path_current_split)
    path_want=path_current_split[0]
    for i in range(len(path_current_split)-1-path_count):
        j=i+1
        path_want=path_want+'/'+path_current_split[j]
    return path_want


def getClassIndices(brestDir_abs_index):

    Breast_Cancer_pytorch_path = get_path(brestDir_abs_index) #得到 Brease_Cancer_pytorch 的绝对路径
    print("路径",Breast_Cancer_pytorch_path)



    json_label_path = os.path.join(Breast_Cancer_pytorch_path,"src_DataPreprocess","zclass_indices.json")




    assert os.path.exists(json_label_path), "not found {}".format(json_label_path)
    with open(json_label_path, 'r', encoding='UTF-8') as f:
        index_to_label = json.loads(f.read())
        label_to_index = dict((name, i) for (i, name) in index_to_label.items())
    return index_to_label, label_to_index

# 训练集和测试机的划分

def read_split_data(root: str, val_rate: float = 0.2):
    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # 遍历文件夹，一个文件夹对应一个类别
    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # 排序，保证顺序一致
    flower_class.sort()
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    val_images_path = []  # 存储验证集的所有图片路径
    val_images_label = []  # 存储验证集图片对应索引信息
    every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
    # 遍历每个文件夹下的文件
    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        # 遍历获取supported支持的所有文件路径
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        # 获取该类别对应的索引
        image_class = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num.append(len(images))
        # 按比例随机采样验证样本
        val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            if img_path in val_path:  # 如果该路径在采样的验证集样本中则存入验证集
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:  # 否则存入训练集
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print("数据集图片：   {}".format(sum(every_class_num)))
    print("训练集数量：   {}. ".format(len(train_images_path)))
    print("验证集数量：   {}.".format(len(val_images_path)))

    plot_image = True
    if plot_image:
        # 绘制每种类别个数柱状图
        plt.bar(range(len(flower_class)), every_class_num, align='center')
        # 将横坐标0,1,2,3,4替换为相应的类别名称
        plt.xticks(range(len(flower_class)), flower_class)
        # 在柱状图上添加数值标签
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        # 设置x坐标
        plt.xlabel('image class')
        # 设置y坐标
        plt.ylabel('number of images')
        # 设置柱状图的标题
        plt.title('flower class distribution')
        plt.show()

    return train_images_path, train_images_label, val_images_path, val_images_label


'''===================图像粘贴函数======================='''


# 把多张图贴在一起，scale代表的是缩放的比例，imgArray是图像的数组 [ ] ,不足的图像采用 np.zeros_like(img) 进行补充
def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                                                None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale,
                                         scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver

def resize_and_letter_box(image, rows, cols):
    """
    Letter box (black bars) a color image (think pan & scan movie shown
    on widescreen) if not same aspect ratio as specified rows and cols.
    :param image: numpy.ndarray((image_rows, image_cols, channels), dtype=numpy.uint8)
    :param rows: int rows of letter boxed image returned
    :param cols: int cols of letter boxed image returned
    :return: numpy.ndarray((rows, cols, channels), dtype=numpy.uint8)
    """
    image_rows, image_cols = image.shape[:2]
    row_ratio = rows / float(image_rows)
    col_ratio = cols / float(image_cols)
    ratio = min(row_ratio, col_ratio)
    image_resized = cv2.resize(image, dsize=(0, 0), fx=ratio, fy=ratio)
    letter_box = np.zeros((int(rows), int(cols), 3)) + 255
    row_start = int((letter_box.shape[0] - image_resized.shape[0]) / 2)
    col_start = int((letter_box.shape[1] - image_resized.shape[1]) / 2)
    letter_box[row_start:row_start + image_resized.shape[0],
    col_start:col_start + image_resized.shape[1]] = image_resized
    return letter_box

# 清除并删除文件夹
def clear_and_create_dir(dir):
    try:
        shutil.rmtree(dir)
        os.mkdir(dir)
    except:
        if os.path.exists(dir) == False:
            os.makedirs(dir)
    print(f"文件夹 {dir} 清除，并重新创建成功！")

# 创建多级目录


def mkdir_multi(path):
    # 判断路径是否存在
    isExists = os.path.exists(path)

    if not isExists:
        # 如果不存在，则创建目录（多层）
        os.makedirs(path)
        print(f'目录:{path},创建成功！')
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        #print('目录已存在！')
        return False



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    device = torch.device(parser.parse_args().device if torch.cuda.is_available() else "cpu")



