from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix, accuracy_score, matthews_corrcoef
import math
import numpy as np
import pandas as pd


def pearson_correlation_by_class(y_true, y_pred, labels):
    pcc_dict = dict()
    for i, label in enumerate(labels):
        label_true = np.array([x[i] for x in y_true])
        label_pred = np.array([x[i] for x in y_pred])
        df = pd.DataFrame({'true': label_true, 'pred' : label_pred})
        pcc_dict[label] = round(df.corr().iloc[0,1], 4)
        #pcc_dict[label] = round(np.corrcoef(label_true.flatten(),label_pred.flatten())[0][1],4)
    df = pd.DataFrame({'true': y_true.flatten(), 'pred' : y_pred.flatten()})
    pcc_dict['overall'] = round(df.corr().iloc[0,1], 4)
    #pcc_dict['overall']= round(np.corrcoef(y_true.flatten(),y_pred.flatten())[0][1],4)
    
    return pcc_dict


def mRNAloc_metric(y_true, y_pred, labels):
    cm_by_class = multilabel_confusion_matrix(y_true, y_pred, labels=labels)
    print('Confusion matrix by class : \n', cm_by_class)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print('Confusion matrix : \n', cm)

    divider_bias = 0.000001  # for not to divide by 0 

    # calculate sen, spe, acc by class
    print('\nsen, spe, acc by class :')
    sen_by_class = []
    spe_by_class = []
    acc_by_class = []
    mcc_by_class = []
    for i in range(len(cm_by_class)):
        bcn = cm_by_class[i]
        tn, fp, fn, tp = bcn[0][0], bcn[0][1], bcn[1][0], bcn[1][1]
        sen = tp / (tp + fn + divider_bias)
        spe = tn / (tn + fp + divider_bias)
        acc = (tp + tn) / (tn + fp + fn + tp + divider_bias)
        mcc = ((tp * tn) - (fp * fn)) / (math.sqrt((tp+fp) * (tp+fn) * (tn+fp) * (tn+fn)) + divider_bias)
        print(labels[i])
        print ('Sen : ', '%.4f'%sen)
        print ('Spe : ', '%.4f'%spe)
        print ('Acc : ', '%.4f'%acc)
        print ('Mcc : ', '%.4f'%mcc)
        sen_by_class.append(sen)
        spe_by_class.append(spe)
        acc_by_class.append(acc)
        mcc_by_class.append(mcc)

    # calculate overall sen, spe, acc 
    # macro
    print('\noverall sen, spe, acc : ')
    macro_sen = np.mean(sen_by_class)
    macro_spe = np.mean(spe_by_class)
    macro_acc = np.mean(acc_by_class)
    print('Macro : \n', 'sen : ', '%4f'%macro_sen, '\n', 'spe : ', '%4f'%macro_spe, '\n', 'acc : ', '%4f'%macro_acc, '\n')

    # weighted
    unique, count = np.unique(y_true, return_counts = True)
    weighted_sen = np.average(sen_by_class, weights = count)
    weighted_spe = np.average(spe_by_class, weights = count)
    weighted_acc = np.average(acc_by_class, weights = count)
    print('Weighted : \n', 'sen : ', '%4f'%weighted_sen, '\n', 'spe : ', '%4f'%weighted_spe, '\n', 'acc : ', '%4f'%weighted_acc, '\n')

    # cal micro mcc, acc 
    mcc = matthews_corrcoef(y_true, y_pred)
    print('\nMCC : ', '%4f'%mcc)
    acc = accuracy_score(y_true, y_pred)
    print('\nACC : ', '%4f'%acc)
    