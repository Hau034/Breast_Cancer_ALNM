from matplotlib import pyplot as plt

from src_utils.utils import clear_and_create_dir

index_to_label = dict({0: "Benign", 1: "Malignant"})
label_to_index = dict((name, i) for (i, name) in index_to_label.items())


def plot_data_loader_image(data_loader, batch_size=8, plotBatchOfNums=4, datasetType="bmode", unNormal=1, save_dir=" "):


    #clear_and_create_dir(save_dir)

    # 解决中文显示问题
    plt.rcParams['font.sans-serif'] = ['SimHei']  # SimHei黑体  FangSong仿宋
    plt.rcParams['axes.unicode_minus'] = False
    # batch_size = data_loader.batch_size
    plot_num = min(batch_size, plotBatchOfNums)



    if datasetType == "bmode":
        print("bmode_img")
        for data in data_loader:
            plt.figure(figsize=(20, 8))
            images, labels = data

            if len(labels) < plot_num:
                plot_num = len(labels)
            for i in range(plot_num):
                # [C, H, W] -> [H, W, C]
                img = images[i].numpy().transpose(1, 2, 0)
                if unNormal == 1:
                    # 反Normalize操作
                    img = (img * [0.5, 0.5, 0.5] + [0.5, 0.5, 0.5]) * 255
                elif unNormal == 2:
                    img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
                else:
                    img = (img * [0.5, 0.5, 0.5] + [0.5, 0.5, 0.5]) * 255
                label = labels[i].item()
                plt.subplot(1, plot_num, i + 1)
                plt.xlabel(index_to_label.get(int(label)))
                plt.xticks([])  # 去掉x轴的刻度
                plt.yticks([])  # 去掉y轴的刻度
                plt.imshow(img.astype('uint8'))
            plt.show()
    elif datasetType == "swe":
        for data in data_loader:
            plt.figure(figsize=(20, 8))
            images, labels = data
            if len(labels) < plot_num:
                plot_num = len(labels)
            for i in range(plot_num):
                # [C, H, W] -> [H, W, C]
                img = images[i].numpy().transpose(1, 2, 0)
                # 反Normalize操作
                if unNormal == 1:
                    # 反Normalize操作
                    img = (img * [0.5, 0.5, 0.5] + [0.5, 0.5, 0.5]) * 255
                elif unNormal == 2:
                    img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
                else:
                    img = (img * [0.5, 0.5, 0.5] + [0.5, 0.5, 0.5]) * 255
                label = labels[i].item()
                plt.subplot(1, plot_num, i + 1)
                plt.xlabel(index_to_label.get(int(label)))
                plt.xticks([])  # 去掉x轴的刻度
                plt.yticks([])  # 去掉y轴的刻度
                plt.imshow(img.astype('uint8'))
            plt.show()
    elif datasetType == "bmode_swe":
        for data in data_loader:
            plt.figure(figsize=(20, 8))
            (bmode_img, swe_img), labels = data
            if len(labels) < plot_num:
                plot_num = len(labels)

            for i in range(plot_num):
                b_img = bmode_img[i].numpy().transpose(1, 2, 0)
                s_img = swe_img[i].numpy().transpose(1, 2, 0)
                label = labels[i].numpy()
                if unNormal == 1:
                    # 反Normalize操作
                    b_img = (b_img * [0.5, 0.5, 0.5] + [0.5, 0.5, 0.5]) * 255
                    s_img = (s_img * [0.5, 0.5, 0.5] + [0.5, 0.5, 0.5]) * 255
                elif unNormal == 2:
                    b_img = (b_img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
                    s_img = (s_img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255

                else:
                    b_img = (b_img * [0.5, 0.5, 0.5] + [0.5, 0.5, 0.5]) * 255
                    s_img = (s_img * [0.5, 0.5, 0.5] + [0.5, 0.5, 0.5]) * 255

                plt.subplot(2, plot_num, i + 1)

                plt.imshow(b_img.astype('uint8'))
                plt.subplot(2, plot_num, i + 1 + plot_num)
                plt.xlabel(index_to_label.get(int(label)))
                plt.xticks([])  # 去掉x轴的刻度
                plt.yticks([])  # 去掉y轴的刻度
                plt.imshow(s_img.astype('uint8'))
            plt.show()
    elif datasetType == "bmode_swe_withInfo":
        clear_and_create_dir(save_dir)
        for index, data in enumerate(data_loader):
            print(f"index:{index}")
            plt.figure(figsize=(20, 8))
            (bmode_img, swe_img, img_name), labels = data

            if len(labels) < plot_num:
                plot_num = len(labels)

            for i in range(plot_num):
                b_img = bmode_img[i].numpy().transpose(1, 2, 0)
                s_img = swe_img[i].numpy().transpose(1, 2, 0)

                label = labels[i].numpy()

                # 反Normalize操作
                if unNormal == 1:
                    # 反Normalize操作
                    b_img = (b_img * [0.5, 0.5, 0.5] + [0.5, 0.5, 0.5]) * 255
                    s_img = (s_img * [0.5, 0.5, 0.5] + [0.5, 0.5, 0.5]) * 255
                elif unNormal == 2:
                    b_img = (b_img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
                    s_img = (s_img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255

                else:
                    b_img = (b_img * [0.5, 0.5, 0.5] + [0.5, 0.5, 0.5]) * 255
                    s_img = (s_img * [0.5, 0.5, 0.5] + [0.5, 0.5, 0.5]) * 255
                plt.subplot(2, plot_num, i + 1)

                plt.imshow(b_img.astype('uint8'))
                plt.subplot(2, plot_num, i + 1 + plot_num)

                plt.xlabel(f"类别     : {index_to_label.get(int(label))}({int(label)})；\n"
                           f"图片名    : {img_name[i]}")
                plt.xticks([])  # 去掉x轴的刻度
                plt.yticks([])  # 去掉y轴的刻度
                plt.imshow(s_img.astype('uint8'))
            if save_dir != " ":
                plt.savefig(save_dir + "batch(" + str(index + 1) + "of" + str(len(data_loader)) + ")" + ".jpg")
                # plt.savefig(save_dir + "batch" + str(index + 1)  + ".jpg")
            # plt.show()
            # plt.savefig("./data_loader/" + str(i) + ".jpg")


def printDataInfo(BATCH_SIZE, train_dataloader, val_dataloader, test_dataloader, Isshuffle):
    print(
        "========================================获取的Dataset信息如下===========================================================================")
    print("Dataset的信息如下：\n ")
    print(
        f"BATCH_SIZE: {BATCH_SIZE}   ,steps_per_epoch: {len(train_dataloader)}   , val_step: {len(val_dataloader)} , val_step: {len(test_dataloader)}  ,Isshuffle: {Isshuffle} ")
    print("=========================================")
    print(f"train_dataloader:     {train_dataloader}")
    print(f"val_dataloader:       {val_dataloader}")
    print(f"test_dataloader:      {test_dataloader}")
    print(
        "=====================================================================================================================================")
