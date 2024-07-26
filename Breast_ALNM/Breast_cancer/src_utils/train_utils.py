import sys

from  src_utils.utils import getClassIndices
import torch
from src_utils.confusion_matrix_utils import ConfusionMatrix_ROC
from tqdm import tqdm
import numpy as np



'''
train: 训练一个epoch
参数：
model       ：训练的模型
optimizer   ：使用的优化器，例如Adam、SGD
data_loader : 训练的 Dataset(train_dataloader,val_dataloader)
device      : 设备，使用 GPU 或者 CUP
epoch       : 当前处于第几个 epoch
'''


def train(dataformat,cutoff,model, optimizer, loss_function, train_dataloader, val_dataloader, device, epoch):


    if dataformat == "bmode" or dataformat == "swe":
        train_cutoff,train_loss, train_acc, train_auc = train_one_epoch(dataformat,cutoff, model, optimizer, loss_function,
                                                           train_dataloader, device, epoch)
        val_cutoff,valid_loss, valid_acc, valid_auc = evaluate_one_epoch(dataformat, train_cutoff,model, loss_function, val_dataloader, device,
                                                              epoch)
    elif dataformat == "bmode_swe":
        train_cutoff,train_loss, train_acc, train_auc = train_one_epoch(dataformat, cutoff,model, optimizer, loss_function,
                                                           train_dataloader, device, epoch)
        val_cutoff,valid_loss, valid_acc, valid_auc = evaluate_one_epoch(dataformat,cutoff, model, loss_function, val_dataloader, device,
                                                              epoch)
    else:
        train_loss = 0.
        train_acc = 0.
        train_auc = 0.
        valid_loss = 0.
        valid_acc = 0.
        valid_auc = 0.
        train_cutoff =0
        val_cutoff = 0

    return train_cutoff,val_cutoff,train_loss, train_acc, train_auc, valid_loss, valid_acc, valid_auc


def test(dataformat, cut_off,model, data_loader, device, epoch):
    if dataformat == "bmode" or dataformat == "swe":
        test_cutoff,test_acc, test_acu = test_one_epoch(dataformat,cut_off,model, data_loader, device, epoch)

    elif dataformat == "bmode_swe":
        test_cutoff,test_acc, test_acu = test_one_epoch(dataformat,cut_off, model, data_loader, device, epoch)
    else:
        test_acc = 0.
        test_acu = 0.
        test_cutoff = 0.
    return test_cutoff,test_acc, test_acu


'''
train_one_epoch: 训练一个epoch
参数：
model       ：训练的模型
optimizer   ：使用的优化器，例如Adam、SGD
data_loader : 训练的 Dataset
device      : 设备，使用 GPU 或者 CUP
epoch       : 当前处于第几个 epoch 
'''


def train_one_epoch(dataformat,cutoff, model, optimizer, loss_function, data_loader, device, epoch):


    pred_view = []
    label_view = []
    pro_view = []


    sample_num = 0  # 累计 总共的样本个数
    data_loader = tqdm(data_loader) # 定义tqdm 进度条
    index_to_label, label_to_index = getClassIndices(brestDir_abs_index=2)  # 自动进行调用获取类别字典
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    labels = [label for _, label in index_to_label.items()] # 标签的获取
    train_confusion = ConfusionMatrix_ROC(num_classes=2, labels=labels) # 训练集混淆矩阵的定义


    # 1.进入训练模式
    model.train()
    # 2.优化器梯度清零
    optimizer.zero_grad()
    # 3.批次加载数据进行训练,有可能是train_loder,val_loader,test_loader
    for step, data in enumerate(data_loader):
        if dataformat == "bmode" or dataformat == "swe":
            # 3.1 解出数据和标签
            images, labels = data
            # 3.2 累计总样本的个数
            sample_num += images.shape[0]
            # 3.3 一个batch的数据经过网络模型得到预测值（还没有经过softmax的两个概率值）
            pred = model(images.to(device))

        elif dataformat == "bmode_swe":
            # 3.1 解出数据
            (bmode_img, swe_img), labels = data
            # 3.2 样本累计
            sample_num += bmode_img.shape[0]
            # 3.3 原始预测结果
            pred = model(bmode_img.to(device), swe_img.to(device))

        else:
            pred = None
            print("参数输入错误，dataformat=bmode 或者 swe 或者 bmod_swe")

        # 3.4 经过softmax 得到归一化的两个概率值(维度0是batch维度，维度1是每个例子的预测值)


        trian_outputs_sf = torch.softmax(pred, dim=1)
        # 3.5 获取截断值cutoff

        # train 和 test 不指定data的类型(使用自动计算的cutoff)，在test中进行指定(test使用train的cutoff)
        # 不是主方法，cutoff 默认的截断值是0.5

        if cutoff!=None:
            train_cutoff = train_confusion.update(data="train",
                                                  cutoff=cutoff,
                                                  preds_sf=trian_outputs_sf.detach().to("cpu").numpy(),
                                                  labels=labels.to("cpu").numpy())
        else:
            # 不指定cutoff,自动计算
            train_cutoff = train_confusion.update(data="",
                                                  cutoff="",
                                                  preds_sf=trian_outputs_sf.detach().to("cpu").numpy(),
                                                  labels=labels.to("cpu").numpy())
        # 3.6 根据截断值cutoff,获取预测的标签的值
        # 2022年11月13日00:51:46 获取当前的截断值,根据截断值去获取正确率

        pred_classes = list(map(lambda x: 1 if x >= train_cutoff else 0, trian_outputs_sf[:,1]))

        pro = (trian_outputs_sf[:,1]).cpu().detach().numpy()

        pred_view.append(pred_classes)
        label_view.append(np.array(labels))
        pro_view.append(pro)


        # 3.7 判断这一个batch的预测值和标签值相等的个数，并累加
        accu_num += np.sum(np.array(pred_classes)== np.array(labels))
        # 3.8 损失的计算，
        loss = loss_function(pred, labels.long().to(device))  # pred 是 [-1, 0] 的概率值，labels 可以是一个标签
        # 3.9 反向传播
        loss.backward()
        # 3.10 累加每一个batch的损失
        accu_loss += loss.detach()  # 使用detach切断梯度，否则不可加起来
        # 3.11 更新训练集混淆矩阵的数值（预测的标签和实际标签）和 计算 AUC 需要的参数 (预测的概率值 和 实际标签)
        train_auc = train_confusion.get_rocAuc()

        if train_auc == None:  # 当第一个batch 全部是同一个标签的时候，计算AUC会返回None，需要在这里增加一个判断
            train_auc = 0.
        # 3.13 更新进度条
        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f},auc: {:.3f}".format(epoch,
                                                                                           accu_loss.item() / (step + 1),
                                                                                           accu_num.item() / sample_num,
                                                                                           train_auc)
        ## 如果损失的值是没有意义的，直接退出训练进程
        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)
        # 3.14 使用优化器对梯度进行更新
        optimizer.step()
        # 3.15 进入下一个epoch，梯度再次清零
        optimizer.zero_grad()


    return train_cutoff,accu_loss.item() / (step + 1), accu_num.item() / sample_num, train_auc


'''
evaluate: 对 dataloader的数据验证
备注：不需要进行梯度更新，但是在这里为了看到验证集的损失值，因此需要一个损失函数计算，但这个损失值并不参与反向传播的过程。
参数：
model       ：训练的模型
optimizer   ：使用的优化器，例如Adam、SGD
data_loader : 训练的 Dataset
device      : 设备，使用 GPU 或者 CUP
epoch       : 当前处于第几个 epoc
'''


@torch.no_grad()
def evaluate_one_epoch(dataformat, cutoff,model, loss_function, data_loader, device, epoch):

    pred_view = []
    label_view = []
    pro_view = []
    cutoff_list = []
    sample_num = 0  # 累计 总共的样本个数
    data_loader = tqdm(data_loader)  # 定义tqdm 进度条
    index_to_label, label_to_index = getClassIndices(brestDir_abs_index=2)  # 自动进行调用获取类别字典
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    labels = [label for _, label in index_to_label.items()]  # 标签的获取
    val_confusion = ConfusionMatrix_ROC(num_classes=2, labels=labels)  # 训练集混淆矩阵的定义

    # 1.进入测试模式
    model.eval()
    # 2.批次加载数据进行验证
    for step, data in enumerate(data_loader):
        if dataformat == "bmode" or dataformat == "swe":
            # 2.1 解出测试数据
            images, labels = data
            # 2.2 样本个数的统计
            sample_num += images.shape[0]
            # 2.3 一个batch的数据经过网络
            pred = model(images.to(device))
        elif dataformat == "bmode_swe":
            # 2.1 解出测试数据
            (bmode_img, swe_img), labels = data
            # 2.2 样本个数的统计
            sample_num += bmode_img.shape[0]
            # 2.3 一个batch的数据经过网络
            pred = model(bmode_img.to(device), swe_img.to(device))
        else:
            pred = None
            print("参数输入错误，dataformat=bmode 或者 swe 或者 bmod_swe")
        # 3.4 经过softmax 得到归一化的两个概率值(维度0是batch维度，维度1是每个例子的预测值)

        val_outputs_sf = torch.softmax(pred, dim=1)
        # 3.5 获取截断值cutoff

        # train 和 test 不指定data的类型(使用自动计算的cutoff)，在test中进行指定(test使用train的cutoff)
        if cutoff!=None:
            val_cutoff = val_confusion.update(data="val",
                                              cutoff=cutoff,  # 使用train_cutoff
                                              preds_sf=val_outputs_sf.detach().to("cpu").numpy(),
                                              labels=labels.to("cpu").numpy())
        else:
            val_cutoff = val_confusion.update(data="",
                                              cutoff="",
                                              preds_sf=val_outputs_sf.detach().to("cpu").numpy(),
                                              labels=labels.to("cpu").numpy())
        #print(val_cutoff)
        val_cutoff = 0.5
        # 3.6 根据截断值cutoff,获取预测的标签的值
        # 2022年11月13日00:51:46 获取当前的截断值,根据截断值去获取正确率
        cutoff_list.append(val_cutoff)
        pred_classes = list(map(lambda x: 1 if x >= val_cutoff else 0, val_outputs_sf[:, 1]))

        pro = (val_outputs_sf[:,1]).cpu().numpy()

        pred_view.append(pred_classes)
        label_view.append(np.array(labels))
        pro_view.append(pro)


        # 3.7 判断这一个batch的预测值和标签值相等的个数，并累加
        accu_num += np.sum(np.array(pred_classes) == np.array(labels))
        # 3.8 损失的计算，
        loss = loss_function(pred, labels.long().to(device))  # pred 是 [-1, 0] 的概率值，labels 可以是一个标签
        # 3.10 累加每一个batch的损失
        accu_loss += loss # 没有梯度，不用detach
        # 3.11 更新训练集混淆矩阵的数值（预测的标签和实际标签）和 计算 AUC 需要的参数 (预测的概率值 和 实际标签)
        val_auc = val_confusion.get_rocAuc()
        if val_auc == None:  # 当第一个batch 全部是同一个标签的时候，计算AUC会返回None，需要在这里增加一个判断
            val_auc = 0.
        # 3.13 更新进度条
        data_loader.desc = "[val epoch {}] loss: {:.3f}, acc: {:.3f},auc: {:.3f}".format(epoch,
                                                                                           accu_loss.item() / (step + 1),
                                                                                           accu_num.item() / sample_num,
                                                                                           val_auc)



    return round(val_cutoff,3),accu_loss.item() / (step + 1), accu_num.item() / sample_num, val_auc


'''
test: 对 dataloader的数据进行测试
备注：因为不需要进行梯度更新，故在这里可以不需要进行损失函数的计算，直接让数据经过网络模型得出结果即可
参数：
model       ：训练的模型
optimizer   ：使用的优化器，例如Adam、SGD
data_loader : 训练的 Dataset
device      : 设备，使用 GPU 或者 CUP
epoch       : 当前处于第几个 epoch
'''


@torch.no_grad()
def test_one_epoch(dataformat,cutoff, model, data_loader, device, epoch):
    pro_view = []
    sample_num = 0  # 累计 总共的样本个数
    data_loader = tqdm(data_loader)  # 定义tqdm 进度条
    index_to_label, label_to_index = getClassIndices(brestDir_abs_index=2)  # 自动进行调用获取类别字典
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    labels = [label for _, label in index_to_label.items()]  # 标签的获取
    test_confusion = ConfusionMatrix_ROC(num_classes=2, labels=labels)  # 训练集混淆矩阵的定义

    # 1.进入测试模式
    model.eval()
    # 2. 对测试的数据一次进行batch的测试
    for step, data in enumerate(data_loader):

        if dataformat == "bmode" or dataformat == "swe":
            # 2.1 解出测试数据
            images, labels = data
            # 2.2 样本个数的统计
            sample_num += images.shape[0]
            # 2.3 一个batch的数据经过网络
            pred = model(images.to(device))
        elif dataformat == "bmode_swe":
            # 2.1 解出测试数据
            (bmode_img, swe_img), labels = data
            # 2.2 样本个数的统计
            sample_num += bmode_img.shape[0]
            # 2.3 一个batch的数据经过网络
            pred = model(bmode_img.to(device), swe_img.to(device))
        else:
            pred = None
            print("参数输入错误，dataformat=bmode 或者 swe 或者 bmod_swe")

        # 3.4 经过softmax 得到归一化的两个概率值(维度0是batch维度，维度1是每个例子的预测值)
        test_outputs_sf = torch.softmax(pred, dim=1)

        pro_view.append((test_outputs_sf[:, 1]).cpu().numpy())

        # 3.5 获取截断值cutoff

        # train 和 test 不指定data的类型(使用自动计算的cutoff)，在test中进行指定(test使用train的cutoff)
        test_cutoff =test_confusion.update(data="test",
                                           cutoff=cutoff,
                                           preds_sf=test_outputs_sf.detach().to("cpu").numpy(),
                                           labels=labels.to("cpu").numpy())
        # 3.6 根据截断值cutoff,获取预测的标签的值
        # 2022年11月13日00:51:46 获取当前的截断值,根据截断值去获取正确率
        pred_classes = list(map(lambda x: 1 if x >= test_cutoff else 0, test_outputs_sf[:, 1]))

        # 3.7 判断这一个batch的预测值和标签值相等的个数，并累加
        # accu_num += torch.eq(np.array(pred_classes), labels.to(device)).sum()
        accu_num += np.sum(np.array(pred_classes) == np.array(labels))
        # 3.8 损失的计算，
        #loss = loss_function(pred, labels.long().to(device))  # pred 是 [-1, 0] 的概率值，labels 可以是一个标签
        # 3.10 累加每一个batch的损失
        #accu_loss += loss.detach()  # 使用detach切断梯度，否则不可加起来
        # 3.11 更新训练集混淆矩阵的数值（预测的标签和实际标签）和 计算 AUC 需要的参数 (预测的概率值 和 实际标签)
        test_auc = test_confusion.get_rocAuc()
        if test_auc == None:
            test_auc = 0.
        data_loader.desc = "[test epoch {}]  acc: {:.3f},auc: {:.3f}".format(epoch, accu_num.item() / sample_num, test_auc)
    return test_cutoff,accu_num.item() / sample_num, test_auc
