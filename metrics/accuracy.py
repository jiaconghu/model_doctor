import numpy as np
import torch


def accuracy(outputs, labels, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = labels.size(0)

        _, pred = outputs.topk(maxk, 1, True, True)  # [batch_size, topk]
        pred = pred.t()  # [topk, batch_size]
        correct = pred.eq(labels.view(1, -1).expand_as(pred))  # [topk, batch_size]

        res = []
        for k in topk:
            correct_k = correct[:k].float().sum()
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class ClassAccuracy:
    def __init__(self, num_classes):
        self.sum = np.zeros(num_classes)
        self.count = np.zeros(num_classes)

    def accuracy(self, outputs, labels):
        _, pred = outputs.max(dim=1)
        correct = pred.eq(labels)

        for b, label in enumerate(labels):
            self.count[label] += 1
            self.sum[label] += correct[b]

    def __str__(self):
        fmtstr = '{}:{:6.2f}'
        avg = (self.sum / self.count) * 100
        result = '\n'.join([fmtstr.format(l, a) for l, a in enumerate(avg)])
        return result
