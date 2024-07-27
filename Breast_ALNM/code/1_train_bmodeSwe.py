import argparse
import math
import os

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from debugConfigFile import getDebugConfig_BmodeSwe
import src_utils.utils as utils
import src_utils.train_utils as train_utils
from src_DataPreprocess.createDataset import generate_ds

'''
Retrieve debugging configuration information
'''
model,\
device,\
data_transform,\
data_version,\
fusion_layer,\
save_weights_path,\
weights,\
trained_weight_path = getDebugConfig_BmodeSwe(trainOrtest="train")


# print(data_version)
# print(fusion_layer)


if model=="MBT_in21k_bmodeswe":
    from models.MBT_model import MBT_in21k_bmodeswe as create_model
else:
    print("error")



def main(args):

    # 清除保存每次训练之后测试图片的路径、清除每次训练之后保存的训练文件
    utils.clear_and_create_dir(args.save_weights_path + "/fusion_layer_" + str(args.fusion_layer))
    tb_writer = SummaryWriter()

    '''
    Get Dataloader
    '''
    train_loader, val_loader, test_loader = generate_ds(datasetType="bmode_swe",
                                                        transforms=data_transform,
                                                        version=data_version,
                                                        isShuffle=True,
                                                        batch_size=8,
                                                        isAugment=args.isDatasetAugment)

    model = create_model(num_classes=args.num_classes, fusion_layer=args.fusion_layer, has_logits=False).to(device)

    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)
        print(model.load_state_dict(weights_dict, strict=False))

    if args.freeze_layers:
        for index, (name, para) in enumerate(model.named_parameters()):

            for name, para in model.named_parameters():
                # 除head, pre_logits外，其他权重全部冻结
                if "head" not in name and "scale" not in name and "out" not in name and "blocks_bmode.11" not in name \
                        and "blocks_swe.11" not in name and "blocks_bmode.10" not in name and "blocks_swe.10" not in name \
                        and "blocks_bmode.9" not in name and "blocks_swe.9" not in name \
                        and "blocks_bmode.8" not in name and "blocks_swe.8" not in name \
                        and "blocks_bmode.7" not in name and "blocks_swe.7" not in name \
                        and "blocks_bmode.6" not in name and "blocks_swe.6" not in name:
                    para.requires_grad_(False)
                else:
                    # print("training {}".format(name))
                    pass

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=5E-5)


    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # loss fuction
    loss_function = torch.nn.CrossEntropyLoss()

    data_format = "bmode_swe"


    for epoch in range(args.epochs):
        train_cutoff,val_cutoff,train_loss, train_acc, train_auc, \
        val_loss, val_acc, val_auc = train_utils.train(dataformat=data_format,
                                                         cutoff=0.5,
                                                         model=model,
                                                         optimizer=optimizer,
                                                         loss_function=loss_function,
                                                         train_dataloader=train_loader,
                                                         val_dataloader=val_loader,
                                                         device=device,
                                                         epoch=epoch)



        scheduler.step()

        '''
        以下函数的调用是可选的

        test
        plot_class_preds

        '''
        test_cutoff,test_acc, test_auc = train_utils.test(dataformat=data_format,
                                                          cut_off=0.5,
                                                          model=model,
                                                          data_loader=test_loader,
                                                          device=device,
                                                          epoch=epoch)

        print("\n\n")

        '''
        plot_class_preds_bmodeSwe 函数的目的
        
        '''
        # add figure into tensorboard
        # fig,test_auc = utils.plot_class_preds(dataformat=data_format,
        #                                  net=model,
        #                                  bmode_images_dir=args.test_image_path_bmode,
        #                                  swe_images_dir=args.test_image_path_swe,
        #                                  transform=data_transform["val"],
        #                                  num_plot=None,
        #                                  device=device)
        #
        # fig.savefig(args.fit_result_path + "/epoch-" + str(epoch) +"-"+ str(round(test_auc,3)) + "_.jpg")
        '''
        在 TesnorBoard显示会存在丢帧的情况，可以使用上一种方法，保存到本地就行代替。
        '''
        # if fig is not None:
        #     tb_writer.add_figure("predictions vs. actuals",
        #                          figure=fig,
        #                          global_step=epoch)

        '''
        Save：
        '''

        print("Save Path：")

        save_path = args.save_weights_path + "/fusion_layer_" + str(args.fusion_layer)
        utils.mkdir_multi(save_path)

        print(save_path)
        torch.save(model.state_dict(), save_path + "/E-{}"
                                                   "-tr_{}_{}"
                                                   "-v_{}_{}"
                                                   "-te_{}_{}"
                                                   "-cf_{}_{}.pth"

                            .format(epoch
                           , round(train_acc, 2), round(train_auc, 2)
                           , round(val_acc, 2), round(val_auc, 2)
                           , round(test_acc, 2), round(test_auc, 2),round(train_cutoff, 3), round(val_cutoff, 3)))




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument('--isDatasetAugment', type=bool, default=False)
    parser.add_argument('--save_model', type=bool, default=True)
    parser.add_argument('--weights', type=str,
                        default=weights)
    parser.add_argument('--save_weights_path', type=str,
                        default=save_weights_path)
    parser.add_argument('--freeze-layers', type=bool, default=True)

    'fusion_layer: Represents different fusion stages. Setting different fusion_layer parameters can control the model to perform feature fusion in early, mid, and late terms.'

    parser.add_argument('--fusion_layer', type=int, default=6)
    opt = parser.parse_args()

    main(opt)
