# 混淆矩阵的绘制
import time

import numpy as np
import pandas as pd
import sklearn
from matplotlib import pyplot as plt
from prettytable import PrettyTable
from sklearn.metrics import roc_curve, roc_auc_score, f1_score
import dataframe_image as dfi

class ConfusionMatrix_ROC(object):
    """
    注意，如果显示的图像不全，是matplotlib版本问题
    本例程使用matplotlib-3.2.1(windows and ubuntu)绘制正常
    需要额外安装prettytable库
    """

    # 初始化使用的参数
    '''
    num_classes : 类别的个数
    labels      : jsion 文件标签

    '''

    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))  # 初始化一个混淆矩阵，用于更新参数
        self.num_classes = num_classes  # 类别
        self.labels = labels  # 标签
        self.y_true = []  # 列表存储，正在运行的值
        # 根据 cutoff 得到的截断值，也就是约登指数
        self.y_pred = []
        self.y_predprob = []  # 预测为正类的概率值
        self.roc_auc = 0.
        self.cutoff = 0.
        self.fpr = np.array([])
        self.tpr = np.array([])
        self.sen=0.
        self.spe=0.
        self.ppv=0.
        self.npv=0.

    # 更新混淆矩阵
    def update(self, data, cutoff, preds_sf, labels):

        # 更新self.matrix。根据cutoff 去计算 self.matrix
        self.y_true.extend(labels)  # 增加批次的数据
        self.y_predprob.extend(preds_sf[:, 1])  # 增加批次的预测值，预测为1的概率
        # 叠加更新，每次清除一下矩阵，以免下次继续叠加
        self.matrix = np.zeros((self.num_classes, self.num_classes))
        # 更新
        # self.cutoff = self.best_confusion_matrix(np.array(self.y_true), np.array(self.y_predprob))

        """
                根据真实值和预测值（预测概率）的向量来计算混淆矩阵和最优的划分阈值

                Args:
                    y_test:真实值
                    y_test_predprob：预测值

                Returns:
                    返回最佳划分阈值和混淆矩阵
                """

        # 根据预测为正类的概率和标签，计算 fpr,tpr,以及threshold
        '''
        每一个 threshold  对应一个 tpr 和 fpr
        '''
        self.fpr, self.tpr, thresholds = roc_curve(self.y_true, self.y_predprob, pos_label=1)
        # 根据 frp和tpr 可以计算 acu 更新auc
        try:
            self.roc_auc = roc_auc_score(np.array(self.y_true).ravel(), np.array(self.y_predprob).ravel())
        except:
            self.roc_auc = 0
        # 找到判断良恶性最优的判别值，即良恶性判别的标准，大于这值的判别为1，小于这个值的判别为0。而非简单的根据概率判别为0或者1
        if data == "test":
            self.cutoff = cutoff
            # print(f"test_cutoff:{cutoff}")
        elif data == "train":
            self.cutoff = cutoff
            # print(f"train_cutoff:{cutoff}")
        elif data == "val":
            self.cutoff = cutoff
            # print(f"val_cutoff:{cutoff}")
        else:
            # 找到是的 youndex = tpr - (1-特异度）= tpr - fpr  最大的那一项的 thresholds 作为判断良恶性的标准
            self.cutoff = self.find_optimal_cutoff(self.tpr, self.fpr, thresholds)  # 根据 fpr 和 tpr 得到约登指数最大，的截断值
            # self.cutoff = round(self.cutoff,3)
        # 得到最好的截断值，根据截断值判断预测的值
        self.y_pred = list(map(lambda x: 1 if x >= self.cutoff else 0, self.y_predprob))
        # 根据预测的值更新混淆矩阵 TP,TN,FP,FN
        for p, t in zip(self.y_pred, self.y_true):
            t = int(t)
            # 行是预测，列是真实标签
            self.matrix[p, t] += 1

        # print(self.y_true)
        # print(self.y_predprob)
        # df = pd.DataFrame({'Predicted_Probability': self.y_predprob,'True_Label': self.y_true})
        #
        # df.to_excel('probs0410.xlsx', index=False)


        return self.cutoff




    def get_rocAuc(self):
        return self.roc_auc

    def get_Cutoff(self):
        return self.cutoff

    def get_senANDspe(self):
        # 更新混淆矩阵之后，返回当前截断值
        TN2 = self.matrix[0, 0]
        TP2 = self.matrix[1, 1]
        FN2 = self.matrix[0, 1]
        FP2 = self.matrix[1, 0]
        print(f"[ TN:{TN2},FN:{FN2}\n  FP:{FP2},TP:{TP2} ]")
        digits = 3  # 保留的小数点
        #Sen Spe
        self.sen, self.spe = round(TP2 / (TP2 + FN2), digits), round(TN2 / (FP2 + TN2), digits)

        # PPV NPV
        self.npv, self.ppv = round(TN2 / (FN2 + TN2), digits), round(TP2 / (TP2 + FP2), digits)
        return self.sen,self.spe,self.ppv,self.npv
    '''
    - tpr: Sen = TP / (TP + FN) TPR越大，则表示挑出的越有可能（是正确的）
    - fpr: Spe = FP / (TN + FP) FPR越大，则表示越不可能
    -  约登指数（Youden Index），是评价预测真实性的方法，假设其假阴性和假阳性的危害性同等意义时，即可应用约登指数。
       约登指数是敏感度与特异度之和减去1，表示预测发现真正的阳性与阴性样本的总能力。 指数越大说明预测的效果越好，真实性越大。
        Younden's = Sen + Spe -1 (敏感度 + 特异性 -1 ）

    函数功能：为了同时最大化敏感度和特异度，我们需要先找到最优的阈值。
    '''

    def find_optimal_cutoff(self, tpr, fpr, threshold):
        optimal_idx = np.argmax(tpr - fpr)  # 最大索引就是约登指数
        optimal_threshold = threshold[optimal_idx]  # 找到约登指数最大的项目的 tpr,fpr 就是敏感度和特异性
        return optimal_threshold

    def plot(self, data="train", save_trainPath="", save_valPath="", save_testPath=""):

        '''================ROC曲线================================='''

        plt.figure(figsize=(10, 4.8))
        plt.subplot(1, 2, 1)
        plt.plot(self.fpr, self.tpr, color='green', lw=2, label=' AUC={0:.3f}'.format(self.roc_auc))
        plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
        plt.legend(loc="lower right")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.title("ROC")
        plt.xlabel('1 - Specificity')
        plt.ylabel('Sensitivity')
        '''==============================================================================='''
        '''========================混淆矩阵=================================='''
        plt.subplot(1, 2, 2)
        matrix = self.matrix
        print(matrix)
        plt.imshow(matrix, cmap=plt.cm.Blues)

        # 设置x轴坐标label
        plt.xticks(range(self.num_classes), self.labels, rotation=45)
        # 设置y轴坐标label
        plt.yticks(range(self.num_classes), self.labels)
        # 显示colorbar
        plt.colorbar()
        plt.xlabel('True Labels')  # 列代表的是真实的案例
        plt.ylabel('Predicted Labels')  # 每一行代表预测的概率
        plt.title('Confusion matrix')

        # 在图中标注数量/概率信息

        TN2 = self.matrix[0, 0]
        TP2 = self.matrix[1, 1]
        FN2 = self.matrix[0, 1]
        FP2 = self.matrix[1, 0]
        thresh = self.matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                # 注意这里的matrix[y, x]不是matrix[x, y]，因为 xticks 跟我们设置的列是一致的
                info = int(matrix[y, x])
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")
        plt.tight_layout()
        if data == "train":
            plt.savefig(save_trainPath)
        elif data == "val":
            plt.savefig(save_valPath)
        else:
            plt.savefig(save_testPath)
        '''======================================================================='''

        '''========================敏感性特异性计算================================='''

        digits = 3  # 保留的小数点

        acc = round((TP2 + TN2) / float(TP2 + TN2 + FP2 + FN2), digits)
        # Sen Spe
        best_recall, best_prec = round(TP2 / (TP2 + FN2), digits), round(TN2 / (FP2 + TN2), digits)

        # PPV NPV
        npv, ppv = round(TN2 / (FN2 + TN2), digits), round(TP2 / (TP2 + FP2), digits)

        # PLR NLR
        plr, nlr = round((TP2 / (TP2 + FN2)) / (FP2 / (FP2 + TN2)), digits), round(
            (FN2 / (TP2 + FN2)) / (TN2 / (FP2 + TN2)), digits)

        # F1值
        y_test_pred = list(map(lambda x: 1 if x >= self.cutoff else 0, self.y_predprob))
        f1 = round(f1_score(self.y_true, y_test_pred), digits)




        # Youden Index
        youden = round(TP2 / (TP2 + FN2) + TN2 / (FP2 + TN2) - 1, digits)

        # MCC
        mcc = round(sklearn.metrics.matthews_corrcoef(self.y_true, self.y_pred), digits)

        # Kappa
        kappa = round(sklearn.metrics.cohen_kappa_score(self.y_pred, self.y_true), digits)
        '''
         从 summary复制过来的
        '''

        # precision, recall, specificity
        table = PrettyTable()
        table.field_names = ["Metric", "Auc", "Acc", "Precision", "Sen(Recall)", "Spec", "PPV", "NPV", "PLR", "NLR",
                             "F1", "Youden", "MCC", "Kappa"]
        # 求出针对每一个类别的信息

        Precision = round(TP2 / (TP2 + FP2), digits) if TP2 + FP2 != 0 else 0.



        '''================================================='''
        pic_df = pd.DataFrame(columns=["AUC", "ACC(%)", "SEN(%)", "SPE(%)"], index=["DAMF"])

        pic_df["AUC"] = self.roc_auc
        pic_df["ACC(%)"] = acc
        pic_df["SEN(%)"] = best_recall
        pic_df["SPE(%)"] = best_prec

        if data == "train":
            table.add_row(
                ["train", self.roc_auc, acc, Precision, best_recall, best_prec, ppv, npv, plr, nlr, f1, youden, mcc,
                 kappa])
        elif data == "val":
            table.add_row(
                ["val", self.roc_auc, acc, Precision, best_recall, best_prec, ppv, npv, plr, nlr, f1, youden, mcc,
                 kappa])
        else:
            table.add_row(
                ["test", self.roc_auc, acc, Precision, best_recall, best_prec, ppv, npv, plr, nlr, f1, youden, mcc,
                 kappa])
        # plt.show()
        print(table)

        return pic_df
