import numpy as np
import torch
from torchvision import transforms
from core.grad_calculate import HookModule


class GradConstraint:

    def __init__(self, module, grad_path, alpha, beta):
        self.module = HookModule(module)
        self.channels = torch.from_numpy(np.load(grad_path)).cuda()
        self.alpha = alpha
        self.beta = beta

    def loss_channel(self, outputs, labels):
        loss = 0

        # high response channel loss
        probs = torch.argsort(-outputs, dim=1)
        labels_ = []
        for i in range(len(labels)):
            if probs[i][0] == labels[i]:
                labels_.append(probs[i][1])  # TP rank2
            else:
                labels_.append(probs[i][0])  # FP rank1
        labels_ = torch.tensor(labels_).cuda()
        nll_loss_ = torch.nn.NLLLoss()(outputs, labels_)
        loss += _loss_channel(channels=self.channels,
                              grads=self.module.grads(outputs=-nll_loss_),
                              labels=labels_,
                              is_high=True)

        # low response channel loss
        nll_loss = torch.nn.NLLLoss()(outputs, labels)
        loss += _loss_channel(channels=self.channels,
                              grads=self.module.grads(outputs=-nll_loss),
                              labels=labels,
                              is_high=False)
        return loss * self.alpha

    def loss_spatial(self, outputs, labels, masks):
        if isinstance(masks, torch.Tensor):  # masks is masks
            nll_loss = torch.nn.NLLLoss()(outputs, labels)
            grads = self.module.grads(outputs=-nll_loss)
            masks = transforms.Resize((grads.shape[2], grads.shape[3]))(masks)
            masks_bg = 1 - masks
            grads_bg = torch.abs(masks_bg * grads)

            loss = grads_bg.sum()
            return loss * self.beta
        else:
            return torch.tensor(0)


def _loss_channel(channels, grads, labels, is_high=True):
    grads = torch.relu(grads)
    channel_grads = torch.sum(grads, dim=(2, 3))  # [batch_size, channels]

    loss = 0
    if is_high:
        for b, l in enumerate(labels):
            loss += (channel_grads[b] * channels[l]).sum()
    else:
        for b, l in enumerate(labels):
            loss += (channel_grads[b] * (1 - channels[l])).sum()
    loss = loss / len(labels)
    return loss
