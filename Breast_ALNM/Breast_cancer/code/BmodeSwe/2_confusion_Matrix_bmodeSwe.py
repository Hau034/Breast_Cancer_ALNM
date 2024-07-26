import argparse
import os

import cv2
import dataframe_image as dfi
import pandas as pd
import torch
from tqdm import tqdm
import src_utils.utils as utils
from src_utils.confusion_matrix_utils import  ConfusionMatrix_ROC
from src_DataPreprocess.createDataset import generate_ds
from debugConfigFile import getDebugConfig_BmodeSwe
index_to_label, label_to_index = utils.getClassIndices(brestDir_abs_index=2)

import warnings
warnings.filterwarnings("ignore")

model,\
device,\
data_transform,\
data_version,\
fusion_layer,\
save_weights_path,\
weights,\
trained_weight_path = getDebugConfig_BmodeSwe(trainOrtest="test")


print(trained_weight_path)

if model=="MBT_in21k_bmodeswe":
    from models.MBT_model import MBT_in21k_bmodeswe as create_model
else:
    print("error")


def main(args):



    if os.path.exists(args.base_result_dir) is False:
        os.makedirs(args.base_result_dir)

    print(data_version)
    train_loader, val_loader, test_loader = generate_ds(datasetType="bmode_swe",
                                                        transforms=data_transform,
                                                        version=data_version,
                                                        batch_size=8,
                                                        isAugment=args.isDatasetAugment)


    model = create_model(num_classes=2, fusion_layer=args.fusion_layer, has_logits=False).to(device)

    model_weight_path = args.model_weight_path
    assert os.path.exists(model_weight_path), "cannot find {} file".format(model_weight_path)
    model.load_state_dict(torch.load(model_weight_path, map_location=device), strict=True)

    model.to(device)

    labels = [label for _, label in index_to_label.items()]

    test_confusion = ConfusionMatrix_ROC(num_classes=2, labels=labels)
    model.eval()
    # 训练集和测试集的图像全部经过网络进行预测



    with torch.no_grad():




        for test_data in tqdm(test_loader):
            (test_images_bmode, test_images_swe), test_labels = test_data
            test_outputs = model(test_images_bmode.to(device), test_images_swe.to(device))
            test_outputs_sf = torch.softmax(test_outputs, dim=1)
            '''
Note！: The cutoff parameter of the update function has two options:

cutoff = " " : This indicates using the Youden index to adaptively calculate the optimal cutoff value from the dataset. The calculated cutoff value will be stored in a local CSV file.
cutoff = "num" : In this example, the value 0.74535376 is the optimal cutoff value adaptively calculated from our test set. We use this cutoff value to compute the confusion matrix. Since this value has already been calculated, it does not need to be saved to a CSV file in this example.
            '''
            test_cutoff = test_confusion.update(data="test",
                                                cutoff=0.74535376,
                                                preds_sf=test_outputs_sf.to("cpu").numpy(),
                                                labels=test_labels.to("cpu").numpy())



    print("\n\n")

    print(f"test_cutoff:{test_cutoff}")
    print("\n\n")

    # 创建一个dataframe 保存 cutoff
    cutoff_df = pd.DataFrame({'test_cutoff':test_cutoff},index=["value"])

    'Obtain the adaptive cutoff value when testing new data, and then save it.'
    # cutoff_df.to_csv("./cutoff.csv")

    pic_df_test = test_confusion.plot(data="test", save_testPath=args.test_roc)
    '''=========================对dataframe的处理=============================='''

    channel_together = pd.concat([
        pic_df_test[['AUC', 'ACC(%)', 'SEN(%)', 'SPE(%)']]
    ], axis=1).fillna(0)

    d = dict(selector="th",
             props=[('text-align', 'center')])

    channel_together_style = channel_together.style.set_properties(**{'width': '4em', 'text-align': 'center'}) \
        .set_table_styles([d]).format("{:.3f}")

    dfi.export(obj=channel_together_style, filename=args.result_table, fontsize=20)
    '''=====================↑====对dataframe的处理====↑=========================='''

    ''''==============================结果图像全部粘贴起来=============================='''





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_weight_path', type=str,
                        default=trained_weight_path)




    parser.add_argument('--base_result_dir', type=str, default='./test_resultImg/BmodeSwe_result')
    parser.add_argument('--result_table', type=str, default=parser.parse_args().base_result_dir + '/result_table.jpg')
    parser.add_argument('--train_roc', type=str, default=parser.parse_args().base_result_dir + "/train_roc.jpg")
    parser.add_argument('--val_roc', type=str, default=parser.parse_args().base_result_dir + "/val_roc.jpg")
    parser.add_argument('--test_roc', type=str, default=parser.parse_args().base_result_dir + "/test_roc.jpg")
    parser.add_argument('--result_pic', type=str,
                        default=parser.parse_args().base_result_dir + "/BmodeSwe_result(vit_model).jpg")
    parser.add_argument('--isDatasetAugment', type=bool, default=False)
    parser.add_argument('--fusion_layer', type=int, default=fusion_layer)
    opt = parser.parse_args()

    main(opt)
