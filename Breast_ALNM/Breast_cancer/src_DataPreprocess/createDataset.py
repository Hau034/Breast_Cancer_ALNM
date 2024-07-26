import argparse
import glob
import os
import random

import numpy as np

seed = 42
# os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"

import pandas as pd
from PIL import Image
import torch

from torchvision import transforms
from src_DataPreprocess.plotDataset import plot_data_loader_image, printDataInfo
from torch.utils.data import DataLoader, Dataset
import src_utils.utils as utils
from src_utils.utils import get_path


# 训练index改成2，查看数据集改成1
index_to_label, label_to_index = utils.getClassIndices(brestDir_abs_index=2)


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


seed_everything(seed)


def init_dataset(args):
    # 读取影像组学的文件并排序
    label_train_data = pd.read_csv(args.dataSet_root_dir + "train.csv", header=0).iloc[:, 1:]
    label_val_data = pd.read_csv(args.dataSet_root_dir + "val.csv", header=0).iloc[:, 1:]
    label_test_data = pd.read_csv(args.dataSet_root_dir + "test.csv", header=0).iloc[:, 1:]
    ## 如果使用增强的图片进行训练


    # 1.读取图像
    # train_Bmode = np.array(glob.glob(args.dataSet_root_dir + "Bmode/train/" + '*.jpg') ) # 得到所有Bmode 图像的路径

    train_Bmode = glob.glob(args.dataSet_root_dir + "Bmode/train/" + '*.jpg')
    train_Swe = glob.glob(args.dataSet_root_dir + "Swe/train/" + '*.jpg')

    # 图像的名字
    # train_img_name = label_train_data["img_name"]
    train_label = label_train_data["label"]

    # 1.读取图像
    val_Bmode = glob.glob(args.dataSet_root_dir + "Bmode/val/" + '*.jpg')  # 得到所有Bmode 图像的路径
    val_Swe = glob.glob(args.dataSet_root_dir + "Swe/val/" + '*.jpg')

    val_label = label_val_data["label"]

    # 2. 读取test文件夹下的图像

    test_Bmode = glob.glob(args.dataSet_root_dir + "Bmode/test/" + '*.jpg')  # 得到所有Bmode 图像的路径
    test_Swe = glob.glob(args.dataSet_root_dir + "Swe/test/" + '*.jpg')

    test_label = label_test_data["label"]

    # 对所有路径进行排序，以便和 csv 文件对应的上


    train_Bmode.sort(key=lambda x: (int(x.split('\\')[1].split('.')[1])))
    train_Swe.sort(key=lambda x: (int(x.split('\\')[1].split('.')[1])))

    val_Bmode.sort(key=lambda x: (int(x.split('\\')[1].split('.')[1])))
    val_Swe.sort(key=lambda x: (int(x.split('\\')[1].split('.')[1])))

    test_Bmode.sort(key=lambda x: (int(x.split('\\')[1].split('.')[1])))
    test_Swe.sort(key=lambda x: (int(x.split('\\')[1].split('.')[1])))

    # 训练集和测试集的图像名字
    train_img_name = [path.split('\\')[1][:-4] for path in train_Bmode]
    val_img_name = [path.split('\\')[1][:-4] for path in val_Bmode]
    test_img_name = [path.split('\\')[1][:-4] for path in test_Bmode]

    print("划分训练集和测试集之后：")
    print(
        f"训练集共有：{len(train_label)}例，其中正样本：{np.sum(train_label < 1)}、负样本：{np.sum(train_label >= 1)};\n"
        f"验证集共有：{len(val_label)}例, 其中正样本：{np.sum(val_label < 1)}、负样本：{np.sum(val_label >= 1)};\n"
        f"测试集共有：{len(test_label)}例, 其中正样本：{np.sum(test_label < 1)}、负样本：{np.sum(test_label >= 1)};")

    return [train_Bmode, val_Bmode, test_Bmode, \
            train_Swe, val_Swe, test_Swe, \
            train_label, val_label, test_label, \
            train_img_name, val_img_name, test_img_name]


class MyDataSet_Bmode(Dataset):
    """自定义数据集"""

    def __init__(self, images_path: list, transform=None):
        self.images_path = images_path
        # images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img_path = self.images_path[item]
        img = Image.open(img_path)
        # RGB为彩色图片，L为灰度图片
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))

        label = img_path.split("\\")[-1].split(".")[0]
        label = 1 if label == "Malignant" else 0

        if self.transform is not None:
            img = self.transform(img)
        return img, label


class MyDataSet_Bmode_swe(Dataset):
    """自定义数据集"""

    def __init__(self, bmode_images_path: list, swe_images_path: list, transform=None):
        self.bmode_images_path = bmode_images_path
        self.swe_images_path = swe_images_path
        # self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.bmode_images_path)

    def __getitem__(self, item):
        img_path = self.bmode_images_path[item]
        bmode_img = Image.open(self.bmode_images_path[item])
        swe_img = Image.open(self.swe_images_path[item])
        # RGB为彩色图片，L为灰度图片
        if (bmode_img.mode != 'RGB') | (swe_img.mode != 'RGB'):
            raise ValueError("image: {} isn't RGB mode.".format(self.bmode_images_path[item]))
        # label = self.images_class[item]

        label = img_path.split("\\")[-1].split(".")[0]
        label = 1 if label == "Malignant" else 0

        if self.transform is not None:
            bmode_img = self.transform(bmode_img)
            swe_img = self.transform(swe_img)

        return [bmode_img, swe_img], label


class MyDataSet_Bmode_swe_withInfo(Dataset):
    """自定义数据集"""

    def __init__(self, bmode_images_path: list, swe_images_path: list, img_name, transform=None):

        self.img_name = img_name
        self.bmode_images_path = bmode_images_path
        self.swe_images_path = swe_images_path
        self.transform = transform

    def __len__(self):
        return len(self.bmode_images_path)

    def __getitem__(self, item):
        img_path = self.bmode_images_path[item]
        bmode_img = Image.open(self.bmode_images_path[item])
        swe_img = Image.open(self.swe_images_path[item])
        img_name = self.img_name[item]

        # RGB为彩色图片，L为灰度图片
        if (bmode_img.mode != 'RGB') | (swe_img.mode != 'RGB'):
            raise ValueError("image: {} isn't RGB mode".format(self.bmode_images_path[item]))
        # label = self.images_class[item]

        label = img_path.split("\\")[-1].split(".")[0]
        label = 1 if label == "Malignant" else 0

        if self.transform is not None:
            bmode_img = self.transform(bmode_img)
            swe_img = self.transform(swe_img)

        return [bmode_img, swe_img, img_name], label


class my_Dateset():
    # 全局默认变量
    IsPrint_DatasetMessage = True

    def __init__(self, datalist, transforms):
        super(my_Dateset, self).__init__()
        # 创建数据集的对象

        self.train_Bmode = datalist[0]
        self.val_Bmode = datalist[1]
        self.test_Bmode = datalist[2]

        self.train_Swe = datalist[3]
        self.val_Swe = datalist[4]
        self.test_Swe = datalist[5]

        self.train_label = datalist[6]
        self.val_label = datalist[7]
        self.test_label = datalist[8]

        self.train_img_name = datalist[9]
        self.val_img_name = datalist[10]
        self.test_img_name = datalist[11]

        # 如果不指定，就使用默认的data_transform
        self.data_transform = transforms

    # 根据参数加载数据集
    def load(self, dataformat="bmode", BATCH_SIZE=16, Isshuffle=False):

        if (dataformat == "bmode"):

            train_data = MyDataSet_Bmode(self.train_Bmode, transform=self.data_transform["train"])
            valid_data = MyDataSet_Bmode(self.val_Bmode, transform=self.data_transform["val"])
            test_data = MyDataSet_Bmode(self.test_Bmode, transform=self.data_transform["val"])

            train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=Isshuffle)
            valid_loader = DataLoader(dataset=valid_data, batch_size=BATCH_SIZE, shuffle=Isshuffle)
            test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=Isshuffle)

            if self.IsPrint_DatasetMessage:
                printDataInfo(BATCH_SIZE, train_loader, valid_loader, test_loader, Isshuffle)
            return train_loader, valid_loader, test_loader
        elif (dataformat == "swe"):
            train_data = MyDataSet_Bmode(self.train_Swe, transform=self.data_transform["train"])
            valid_data = MyDataSet_Bmode(self.val_Swe, transform=self.data_transform["val"])
            test_data = MyDataSet_Bmode(self.test_Swe, transform=self.data_transform["val"])

            train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=Isshuffle)
            valid_loader = DataLoader(dataset=valid_data, batch_size=BATCH_SIZE, shuffle=Isshuffle)
            test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=Isshuffle)

            if self.IsPrint_DatasetMessage:
                printDataInfo(BATCH_SIZE, train_loader, valid_loader, test_loader, Isshuffle)
            return train_loader, valid_loader, test_loader

        elif (dataformat == "bmode_swe"):

            train_data = MyDataSet_Bmode_swe(self.train_Bmode, self.train_Swe, transform=self.data_transform["train"])
            valid_data = MyDataSet_Bmode_swe(self.val_Bmode, self.val_Swe, transform=self.data_transform["val"])
            test_data = MyDataSet_Bmode_swe(self.test_Bmode, self.test_Swe, transform=self.data_transform["val"])

            train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=Isshuffle)
            valid_loader = DataLoader(dataset=valid_data, batch_size=BATCH_SIZE, shuffle=Isshuffle)
            test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=Isshuffle)

            if self.IsPrint_DatasetMessage:
                printDataInfo(BATCH_SIZE, train_loader, valid_loader, test_loader, Isshuffle)
            return train_loader, valid_loader, test_loader


        elif (dataformat == "bmode_swe_withInfo"):

            train_data = MyDataSet_Bmode_swe_withInfo(self.train_Bmode, self.train_Swe, self.train_img_name,
                                                      self.data_transform["train"])
            valid_data = MyDataSet_Bmode_swe_withInfo(self.val_Bmode, self.val_Swe, self.val_img_name,
                                                      self.data_transform["val"])
            test_data = MyDataSet_Bmode_swe_withInfo(self.test_Bmode, self.test_Swe, self.test_img_name,
                                                     self.data_transform["val"])

            train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=Isshuffle)
            valid_loader = DataLoader(dataset=valid_data, batch_size=BATCH_SIZE, shuffle=Isshuffle)
            test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=Isshuffle)

            if self.IsPrint_DatasetMessage:
                printDataInfo(BATCH_SIZE, train_loader, valid_loader, test_loader, Isshuffle)
            return train_loader, valid_loader, test_loader



        else:
            print("!!!!!!!!!!!!!!!!!!!参数有误请重新输入!!!!!!!!!!!!!!!!!!！")
            pass

#from src_utils.utils import get_path
def generate_ds(datasetType="bmode", version ="version_normal",transforms="", batch_size=8, isShuffle=False, isAugment=False):

    # 获取工程相对路径 训练改成2，查看数据集1
    Brease_Cancer_pytorch_path = get_path(2)  # 得到 Brease_Cancer_pytorch 的绝对路径
    parser = argparse.ArgumentParser()
    '''======================================='''
    # 训练集数据和测试机数据路径

    parser.add_argument('--dataSet_root_dir', type=str,
                        default=Brease_Cancer_pytorch_path + "/dataset/TrainTestData/" + version + "/")

    '''
   dataformat: 
       1. bmode
       2. swe
       3. radiomics_bmode_swe_withInfo
   '''
    parser.add_argument('--dataformat', type=str, default=datasetType)  # 使用的数据
    opt = parser.parse_args()


    dataset = my_Dateset(init_dataset(opt), transforms=transforms)  # 创建数据集的对象
    '''==========================================训练/测试数据集 显示 开始====================================================='''

    # 加载数据集 [bmode,swe],label -batch
    train_loader, val_loader, test_loader = dataset.load(dataformat=opt.dataformat, BATCH_SIZE=batch_size,
                                                         Isshuffle=isShuffle)

    '''==========================================训练/测试数据集 显示 结束====================================================='''
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    '''
    dataformat:
    1. bmode
    2. swe
    3. bmode_swe
    4. bmode_swe_withInfo

    '''
    '''
        unNormal的含义，默认参数为1
        1. 代表  img = (img * [0.5, 0.5, 0.5] + [0.5, 0.5, 0.5]) * 255
        2. 代表  img = (img * [0.485, 0.456, 0.406] + [0.229, 0.224, 0.225]) * 255
    '''
    data_transform = {
        "train": transforms.Compose([transforms.Resize((224, 224)),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize((224, 224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    #data_version = "version_normal"
    # 新版本 version_nromal_1101
    data_version = "version_normal_240318_train432/random_seed_5"


    train_loader, val_loader, test_loader = generate_ds(datasetType="bmode_swe_withInfo",
                                                        version=data_version,
                                                        transforms=data_transform,
                                                        batch_size=4,
                                                        isShuffle=False,
                                                        isAugment=False)

    print("打开文件夹：./data_loader/")
    os.startfile(os.path.abspath("./data_loader/"))

    print(f"train_loader :{len(train_loader)}")
    plot_data_loader_image(train_loader, batch_size=4, plotBatchOfNums=4, datasetType="bmode_swe_withInfo", unNormal=2,
                           save_dir="./data_loader/" +data_version + "/train/")
    print(f"val_loader :{len(val_loader)}")
    plot_data_loader_image(val_loader, batch_size=4, plotBatchOfNums=4, datasetType="bmode_swe_withInfo", unNormal=2,
                           save_dir="./data_loader/" +data_version + "/val/")
    print(f"test_loader :{len(test_loader)}")
    plot_data_loader_image(test_loader, batch_size=4, plotBatchOfNums=4, datasetType="bmode_swe_withInfo", unNormal=2,
                           save_dir="./data_loader/" +data_version + "/test/")
