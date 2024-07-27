import numpy as np
import pandas as pd
import sklearn
from matplotlib import pyplot as plt
from prettytable import PrettyTable
from sklearn.metrics import roc_curve, roc_auc_score, f1_score

'''
参数1： y_ture 标签
参数2： y_predprob 预测的概率值，是一个一维向量
参数3： cut_off 截断值，如果为None则在代码中自动计算截断值，否则使用cut_off = 0.5 作为截断值
'''
def plot_table(y_true,y_predprob,cut_off,row_name):
    
    digits = 3  # 保留的小数点
    confusion_matrix = np.zeros((2, 2))  # 混淆矩阵初始化
    
    fpr, tpr, thresholds = roc_curve(y_true, y_predprob, pos_label=1)
    if cut_off==None:
        optimal_idx = np.argmax(tpr - fpr)  # 最大索引就是约登指数
        cut_off = thresholds[optimal_idx]  # 找到约登指数最大的项目的 tpr,fpr 就是敏感度和特异性
    else:
        cut_off = 0.5
    

    roc_auc = roc_auc_score(y_true.ravel(), y_predprob.ravel())
    # 得到预测的值 和混淆矩阵
    y_test_pred = list(map(lambda x: 1 if x >= cut_off else 0, y_predprob))
    for p, t in zip(y_test_pred, y_true):
        t = int(t)
        # 行是预测，列是真实标签
        confusion_matrix[p, t] += 1


    '''========================混淆矩阵=================================='''
    # plt.subplot(1, 1, 1)
    # plt.imshow(confusion_matrix, cmap=plt.cm.Blues)
    #
    # # 设置x轴坐标label
    # plt.xticks(range(2), ["Benign","Malignant"], rotation=45)
    # # 设置y轴坐标label
    # plt.yticks(range(2), ["Benign","Malignant"])
    # # 显示colorbar
    # plt.colorbar()
    # plt.xlabel('True Labels')  # 列代表的是真实的案例
    # plt.ylabel('Predicted Labels')  # 每一行代表预测的概率
    # plt.title('Confusion matrix')
    #
    # thresh = confusion_matrix.max() / 2
    #
    # for x in range(2):
    #     for y in range(2):
    #         # 注意这里的confusion_matrix[y, x]不是confusion_matrix[x, y]，因为 xticks 跟我们设置的列是一致的
    #         info = int(confusion_matrix[y, x])
    #         plt.text(x, y, info,
    #                  verticalalignment='center',
    #                  horizontalalignment='center',
    #                  color="white" if info > thresh else "black")
    # plt.tight_layout()

    '''======================================================================='''

    '''========================敏感性特异性计算================================='''

    # 在图中标注数量/概率信息

    TN2 = confusion_matrix[0, 0]
    TP2 = confusion_matrix[1, 1]
    FN2 = confusion_matrix[0, 1]
    FP2 = confusion_matrix[1, 0]


    acc = round((TP2 + TN2) / float(TP2 + TN2 + FP2 + FN2), digits)
    # Sen Spe
    best_recall, best_prec = round(TP2 / (TP2 + FN2), digits), round(TN2 / (FP2 + TN2), digits)

    # PPV NPV
    npv, ppv = round(TN2 / (FN2 + TN2), digits), round(TP2 / (TP2 + FP2), digits)

    # PLR NLR

    plr, nlr = round((TP2 / (TP2 + FN2)) / (FP2 / (FP2 + TN2)), digits), round(
        (FN2 / (TP2 + FN2)) / (TN2 / (FP2 + TN2)), digits)

    # F1值
    f1 = round(f1_score(y_true, y_test_pred), digits)

    # Youden Index
    youden = round(TP2 / (TP2 + FN2) + TN2 / (FP2 + TN2) - 1, digits)

    # MCC
    mcc = round(sklearn.metrics.matthews_corrcoef(y_true, y_test_pred), digits)

    # Kappa
    kappa = round(sklearn.metrics.cohen_kappa_score(y_test_pred, y_true), digits)
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
    pic_df = pd.DataFrame(columns=["AUC", "ACC", "SEN", "SPE","PPV","NPV"], index=[row_name])

    pic_df["AUC"] = roc_auc
    pic_df["ACC"] = acc
    pic_df["SEN"] = best_recall
    pic_df["SPE"] = best_prec
    pic_df["PPV"] = ppv
    pic_df["NPV"] = npv



    table.add_row([row_name, roc_auc, acc, Precision, best_recall, best_prec, ppv, npv, plr, nlr, f1, youden, mcc,kappa])
    print(table)
    
    return pic_df,confusion_matrix