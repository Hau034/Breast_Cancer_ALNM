import os
import shutil
import sys

from sklearn.metrics import roc_auc_score

from src_utils.utils import getClassIndices,clear_and_create_dir
import torch
import numpy as np
from PIL import Image
import pandas as pd
import glob
import matplotlib.pyplot as plt


def plot_class_preds(dataformat: str,
                     data:str,
                     cutoff,
                     net,
                     bmode_images_dir: str,
                     swe_images_dir: str,
                     transform,
                     num_plot: int = 5,
                     device="cpu"):
    # 自动进行调用获取类别字典
    index_to_label, label_to_index = getClassIndices(brestDir_abs_index=0)
    # 判断是否存在测试集的图片
    if not os.path.exists(bmode_images_dir):
        print("not found {} path, ignore add figure.".format(bmode_images_dir))
        return None
    if not os.path.exists(swe_images_dir):
        print("not found {} path, ignore add figure.".format(swe_images_dir))
        return None
    # 获取标签，直接从图片中获取就可以了
    all_testImg_path_bmode = glob.glob(bmode_images_dir + "/*.jpg")
    all_testImg_path_bmode.sort(key=lambda x: (x.split('\\')[1].split('.')[0]))

    all_testImg_path_swe = glob.glob(swe_images_dir + "/*.jpg")
    all_testImg_path_swe.sort(key=lambda x: (x.split('\\')[1].split('.')[0]))


    #print(all_testImg_path_bmode[0].split("/")[-1].split(".")[2].split("(")[1].split(")")[0].split("-")[0])

    ID1 = [path.split("/")[-1].split(".")[2].split("(")[1].split(")")[0].split("-")[0] for path in all_testImg_path_bmode]
    ID2 = [path.split("/")[-1].split(".")[2].split("(")[1].split(")")[0].split("-")[1] for path in all_testImg_path_bmode]

    pred_df = pd.DataFrame({'ID1': ID1,
                            'ID2': ID2})




    # print(f"测试集的长度：{len(all_testImg_path)}")

    label = []  #
    label_info = []
    for bmode_path, swe_path in zip(all_testImg_path_bmode, all_testImg_path_swe):
        label.append(label_to_index.get(bmode_path.split("\\")[1].split(".")[0]))
        class_name = bmode_path.split("\\")[1].split(".")[0]
        label_info.append([bmode_path, swe_path, class_name])

    if len(label_info) == 0:
        return None

    # get first num_plot info

    if num_plot == None:
        pass
    else:
        label_info = label_info[:num_plot]

    num_imgs = len(label_info)
    images_bmode = []
    images_swe = []
    labels = []
    image_name_list = []

    for img_path_bmode, img_path_swe, class_name in label_info:
        image_name_list.append(img_path_bmode.split("\\")[1].split(".jpg")[0])
        #print(img_name)
        # read img
        img_bmode = Image.open(img_path_bmode).convert("RGB")
        img_swe = Image.open(img_path_swe).convert("RGB")

        label_index = int(label_to_index[class_name])

        # preprocessing
        img_bmode = transform(img_bmode)
        img_swe = transform(img_swe)
        images_bmode.append(img_bmode)
        images_swe.append(img_swe)

        labels.append(label_index)

    # batching images
    images_bmode = torch.stack(images_bmode, dim=0).to(device)
    images_swe = torch.stack(images_swe, dim=0).to(device)

    # 进入预测模式
    net.eval()



    with torch.no_grad():
        if dataformat == "bmode":
            output = net(images_bmode)
        elif dataformat == "swe":
            output = net(images_swe)
        elif dataformat == "bmode_swe":
            output = net(images_bmode, images_swe)
        else:
            output = None

        outputs_sf = torch.softmax(output, dim=1)

        # 得到最好的极端值，根据截断值就散预测的值
        preds = list(map(lambda x: 1 if x >= cutoff else 0, outputs_sf[:,1]))

        cutoff_list = preds.copy()

        cutoff_list = [cutoff for _ in cutoff_list]

        probs = outputs_sf.cpu().numpy()[:, 1]



        pred_df.insert(loc=len(pred_df.columns), column="outputs_sf[0]", value=np.round(outputs_sf.cpu().numpy()[:, 0], 4))
        pred_df.insert(loc=len(pred_df.columns), column="outputs_sf[1]", value=np.round(outputs_sf.cpu().numpy()[:, 1], 4))
        pred_df.insert(loc=len(pred_df.columns), column="cutoff", value=np.array(cutoff_list))
        pred_df.insert(loc=len(pred_df.columns), column="preds", value=np.array(preds))
        pred_df.insert(loc=len(pred_df.columns), column="ture_labels", value=np.array(labels))

        print(pred_df)


        if dataformat == "bmode":
            if data=="train":
                print("bmode_tarin输出")
                pred_df.to_csv("./train_bmode_pred_df.csv",index=False)
            elif data == "val":
                print("bmode_val输出")
                pred_df.to_csv("./val_bmode_pred_df.csv", index=False)
            else:
                print("bmode_test输出")
                pred_df.to_csv("./test_bmode_pred_df.csv", index=False)
        elif dataformat == "swe":
            if data == "train":
                print("swe_tarin输出")
                pred_df.to_csv("./train_swe_pred_df.csv", index=False)
            elif data == "val":
                print("swe_val输出")
                pred_df.to_csv("./val_swe_pred_df.csv", index=False)
            else:
                print("swe_test输出")
                pred_df.to_csv("./test_swe_pred_df.csv", index=False)
        elif dataformat == "bmode_swe":
            if data == "train":
                print("bmodeSwe_tarin输出")
                pred_df.to_csv("./train_bmodeSwe_pred_df.csv", index=False)
            elif data == "val":
                print("bmodeSwe_val输出")
                pred_df.to_csv("./val_bmodeSwe_pred_df.csv", index=False)
            else:
                print("bmodeSwe_test输出")
                pred_df.to_csv("./test_bmodeSwe_pred_df.csv", index=False)
        else:
            pass

    # 直接计算 AUC

    auc_compute = roc_auc_score(np.array(labels), outputs_sf.cpu().numpy()[:, 1])





    # width, height
    total_column = 6
    total_row = num_imgs // total_column if num_imgs % total_column == 0 else num_imgs // total_column + 1

    if dataformat == "bmode" or dataformat == "swe":



        if dataformat == "bmode":
            images = images_bmode
        else:
            images = images_swe

        fig = plt.figure(figsize=(total_column * 3, total_row * 3.5), dpi=100)
        for i in range(num_imgs):
            if num_imgs > 6:

                ax = fig.add_subplot(total_row, total_column, i + 1, xticks=[], yticks=[])
            else:
                # 1：子图共1行，num_imgs:子图共num_imgs列，当前绘制第i+1个子图
                ax = fig.add_subplot(1, num_imgs, i + 1, xticks=[], yticks=[])

            # CHW -> HWC
            npimg = images[i].cpu().numpy().transpose(1, 2, 0)

            # 将图像还原至标准化之前
            # mean:[0.485, 0.456, 0.406], std:[0.229, 0.224, 0.225]
            npimg = (npimg * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            plt.imshow(npimg.astype('uint8'))




            title = "pred:{}\n prob {:.3f}\ncutoff:{:.3f}\n(label: {})\n{}".format(
                index_to_label[str(preds[i])],  # predict class
                probs[i],  # predict probability
                cutoff,
                index_to_label[str(labels[i])], # true class
                image_name_list[i]
            )
            ax.set_title(title, color=("green" if preds[i] == labels[i] else "red"))


        # 计算 acc
        acc = round((sum(np.array(preds) == np.array(labels)) /  len(labels)),3)
        print(f"ACC:{acc},AUC:{round(auc_compute, 3)},plot_testImg_pred,img_len:{num_imgs}")







    # 多模态打印的图
    else:

        # 计算 acc
        acc = round((sum(np.array(preds) == np.array(labels)) / len(labels)), 3)
        print(f"ACC:{acc},AUC:{round(auc_compute, 3)},plot_testImg_pred,img_len:{num_imgs * 2}")
        fig = plt.figure(figsize=(total_column * 3, total_row * 8), dpi=100)

        # 重新创建一个文件夹用于存放划分错误的数据
        worng_dir = bmode_images_dir + "/wrong/"
        correct_dir = bmode_images_dir + "/correct/"
        clear_and_create_dir(worng_dir)
        clear_and_create_dir(correct_dir)

        # 保存整个图
        for i in range(num_imgs):
            ax = fig.add_subplot(total_row * 2, total_column, 2 * i + 1, xticks=[], yticks=[])
            # CHW -> HWC
            npimg_bmode = images_bmode[i].cpu().numpy().transpose(1, 2, 0)
            # 将图像还原至标准化之前
            # mean:[0.485, 0.456, 0.406], std:[0.229, 0.224, 0.225]
            npimg_bmode = (npimg_bmode * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255



            plt.imshow(npimg_bmode.astype('uint8'))

            '''swe 图像'''

            ax = fig.add_subplot(total_row * 2, total_column, 2 * i + 2, xticks=[], yticks=[])

            npimg_swe = images_swe[i].cpu().numpy().transpose(1, 2, 0)
            npimg_swe = (npimg_swe * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255

            plt.imshow(npimg_swe.astype('uint8'))


            title = "pred:{}\n prob {:.4f}\ncutoff:{:.4f}\n(label: {})\n{}".format(
                index_to_label[str(preds[i])],  # predict class
                probs[i],  # predict probability
                cutoff,
                index_to_label[str(labels[i])],
                image_name_list[i]   # true class
            )
            # #ax2.set_title(title, color=("green" if preds[i] == labels[i] else "red"))
            # ax2.set_title(title1, color=("green" if preds[i] == labels[i] else "red"))


            ax.set_title(title, color=("green" if preds[i] == labels[i] else "red"))

        wrong_df = pd.DataFrame(columns=['ID1', "ID2", 'label'], index=None)
        correct_df = pd.DataFrame(columns=['ID1', "ID2", 'label'], index=None)
        # 先保存错误的子图吧
        for i in range(num_imgs):

            # CHW -> HWC
            npimg_bmode = images_bmode[i].cpu().numpy().transpose(1, 2, 0)
            # 将图像还原至标准化之前
            # mean:[0.485, 0.456, 0.406], std:[0.229, 0.224, 0.225]
            npimg_bmode = (npimg_bmode * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            '''swe 图像'''
            npimg_swe = images_swe[i].cpu().numpy().transpose(1, 2, 0)
            npimg_swe = (npimg_swe * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            fig1 = plt.figure(figsize=(2 * 3, 1 * 3.5), dpi=100)

            # bmode 图像
            fig1.add_subplot(1, 2, 1, xticks=[], yticks=[])
            plt.imshow(npimg_bmode.astype('uint8'))

            '''swe 图像'''

            ax2 = fig1.add_subplot(1, 2, 2, xticks=[], yticks=[])

            plt.imshow(npimg_swe.astype('uint8'))



    return fig, acc,auc_compute