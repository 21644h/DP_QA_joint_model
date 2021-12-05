import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, precision_recall_fscore_support


def uas_las(outputs):
    uas = 0
    las = 0
    golden_cnt = 0
    bi_pred = 0
    multi_pred = 0
    gold_no_adj = 0
    link_no_adj = 0
    rela_no_adj = 0
    for output in outputs:
        relation_pred = output[0].cpu()
        relation_prob = F.softmax(torch.Tensor(relation_pred), dim=1)
        relation_pred_label = torch.argmax(relation_prob, dim=1)
        type_prob = None
        type_pred_label = None
        if len(output[1]):
            type_prob = F.softmax(torch.Tensor(output[1].cpu()), dim=1)
            type_pred_label = torch.argmax(type_prob, dim=1)
        adjacent_matrix = output[2].cpu()
        k = 0
        for i in range(relation_pred_label.shape[0]):
            no_relation = True
            for j in range(relation_pred_label.shape[0]):
                if adjacent_matrix[i][j]!=16:
                    if abs(i-j)>1:             #
                        gold_no_adj = gold_no_adj + 1
                    golden_cnt = golden_cnt + 1
                    no_relation = False
                    if relation_pred_label[i]==j:
                        bi_pred = bi_pred + 1
                        if abs(i - j) > 1:
                            link_no_adj = link_no_adj + 1
                        if type_pred_label[k]==adjacent_matrix[i][j]:
                            multi_pred = multi_pred + 1
                            if abs(i - j) > 1:
                                rela_no_adj = rela_no_adj + 1

            if relation_pred_label[i]<relation_pred_label.shape[0]:
                k =k + 1

            if no_relation:
                golden_cnt = golden_cnt + 1
                if relation_pred_label[i]>=relation_pred_label.shape[0]:
                    bi_pred = bi_pred + 1
                    multi_pred = multi_pred + 1

    uas = float(bi_pred)/golden_cnt
    las = float(multi_pred)/golden_cnt

    print('total no adj:', gold_no_adj)
    print('link no adj:', link_no_adj)
    print('rela no adj:', rela_no_adj)

    print("Link prediction f1 score:",uas)
    print("Link&type prediction f1 score:",las)
    return uas, las

def uas_las_adj(outputs):
    uas = 0
    las = 0
    golden_cnt = 0
    bi_pred = 0
    multi_pred = 0
    for output in outputs:
        relation_pred = output[0].cpu()
        sigmoid = torch.nn.Sigmoid()
        # relation_prob = torch.nn.Sigmoid(relation_pred)
        # relation_pred_label = torch.argmax(relation_prob, dim=2)
        type_prob = None
        type_pred_label = None
        if len(output[1]):
            type_prob = F.softmax(torch.Tensor(output[1].cpu()), dim=1)
            type_pred_label = torch.argmax(type_prob, dim=1)
        adjacent_matrix = output[2].cpu()
        pred_type_adj = torch.zeros(relation_pred.shape,dtype=torch.float32)
        k = 0
        for i in range(relation_pred.shape[0]):
            relation_prob = sigmoid(relation_pred[i])
            for j in range(relation_pred.shape[0]):
                if relation_prob[j]>0.5:
                    pred_type_adj[i][j]=type_pred_label[k]
                    k = k + 1
        for i in range(relation_pred.shape[0]):
            no_relation = True
            pred_no_relation = True
            relation_prob = sigmoid(relation_pred[i])
            for j in range(relation_pred.shape[0]):
                if adjacent_matrix[i][j]!=16:
                    golden_cnt = golden_cnt + 1
                    no_relation = False
                    if relation_prob[j]>0.5:
                        bi_pred = bi_pred + 1
                        if pred_type_adj[i][j]==adjacent_matrix[i][j]:
                            multi_pred = multi_pred + 1
                if relation_prob[j]>0.5:
                    pred_no_relation = False

            if no_relation:
                golden_cnt = golden_cnt + 1
                if pred_no_relation:
                    bi_pred = bi_pred + 1
                    multi_pred = multi_pred + 1

    uas = float(bi_pred)/golden_cnt
    las = float(multi_pred)/golden_cnt

    print("Link prediction f1 score:",uas)
    print("Link&type prediction f1 score:",las)
    return uas, las