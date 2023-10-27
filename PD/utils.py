import torch


def cal_acc_recall(preds, target, num_classes):
    pred_classes = torch.argmax(preds, dim=1)

    # 计算准确率
    correct = (pred_classes == target).sum().item()
    accuracy = correct / preds.size(0)
    # print("准确率:", accuracy)

    # 计算召回率
    recall = []
    for c in range(num_classes):
        true_positives = ((pred_classes == c) & (target == c)).sum().item()
        total_positives = (target == c).sum().item()
        if total_positives > 0:
            recall.append(true_positives / total_positives)
        else:
            recall.append(0.0)
    # print("召回率:", recall)
    return accuracy, recall