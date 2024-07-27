import argparse

import pandas as pd
import torch
from debugConfigFile import getDebugConfig_BmodeSwe
import src_utils.utils as utils
import src_utils.plot_dataset_pred_utils as plot_dataset_pred_utils
'''
获取调试配置信息

'''
model,\
device,\
data_transform, \
data_version, \
fusion_layer,\
save_weights_path,\
weights,\
trained_weight_path = getDebugConfig_BmodeSwe(trainOrtest="test")


if model=="MBT_in21k_bmodeswe":
    from models.MBT_model import MBT_in21k_bmodeswe as create_model
else:
    print("error")


def main(data_format,fusion_layer,version_data,weight_path):


    parser = argparse.ArgumentParser()

    # The file location of the test set, modify the path here. This path is only for the test example; for actual testing, please change it to ../../dataset/TrainTestData/" + version_data + "/Bmode/test.

    parser.add_argument('--test_image_path_bmode', type=str,
                        default="./dataset/TestData/Bmode")
    parser.add_argument('--test_image_path_swe', type=str,
                        default="./dataset/TestData/Swe")



    parser.add_argument('--model_weight_path', type=str,
                        default=weight_path)

    model = create_model(num_classes=2, fusion_layer=fusion_layer, has_logits=False).to(device)

    # 加载模型参数
    model.load_state_dict(torch.load(parser.parse_args().model_weight_path, map_location=device))

    args = parser.parse_args()




    return data_format,data_transform,args,model,device




if __name__ == '__main__':


    '''
    运行代码之前必选的几个选项：
    参数：
        model_weight_path --- (模型的保存路径，不要混用)
        
         
    
    '''
    data_format = "bmode_swe"
    base_dir = "./test_result"
    utils.clear_and_create_dir(base_dir)

    cutoff = pd.read_csv("./cutoff.csv")


    train_cutoff = cutoff.iloc[0,1]
    val_cutoff = cutoff.iloc[0,2]
    test_cutoff = cutoff.iloc[0,3]



   #生成pred_df文件

    data_format, data_transform, args, model, device = main(data_format=data_format,
                                                            version_data=data_version,
                                                            fusion_layer=int(fusion_layer),
                                                            weight_path=trained_weight_path)



    test_fig, test_acc,test_auc = plot_dataset_pred_utils.plot_class_preds(dataformat=data_format,
                                                                            data="test",
                                                                            cutoff=test_cutoff,
                                                                            net=model,
                                                                            bmode_images_dir=args.test_image_path_bmode,
                                                                            swe_images_dir=args.test_image_path_swe,
                                                                            transform=data_transform["val"],
                                                                            num_plot=None,
                                                                            device=device)
    # #
    test_fig.savefig(base_dir + "/test_result_acc_auc " + str(round(test_acc, 3)) + "_"+ str(round(test_auc, 3)) + "_.jpg")
