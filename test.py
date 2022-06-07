import collections
import os
import argparse
import time

import torch
from torch import nn

import loaders
import models
import metrics
from utils.train_util import AverageMeter, ProgressMeter


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model_name', default='', type=str, help='model name')
    parser.add_argument('--data_name', default='', type=str, help='data name')
    parser.add_argument('--in_channels', default='', type=int, help='in channels')
    parser.add_argument('--num_classes', default='', type=int, help='num classes')
    parser.add_argument('--model_path', default='', type=str, help='model path')
    parser.add_argument('--data_path', default='', type=str, help='data path')
    parser.add_argument('--device_index', default='0', type=str, help='device index')
    args = parser.parse_args()

    # ----------------------------------------
    # basic configuration
    # ----------------------------------------
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_index
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print('-' * 50)
    print('TEST ON:', device)
    print('MODEL PATH:', args.model_path)
    print('-' * 50)

    # ----------------------------------------
    # trainer configuration
    # ----------------------------------------
    model = models.load_model(model_name=args.model_name, num_classes=args.num_classes)
    model.load_state_dict(torch.load(args.model_path))
    model.to(device)
    model.eval()

    test_loader = loaders.load_data(args.data_name, args.data_path, data_type='test')

    criterion = nn.CrossEntropyLoss()

    # ----------------------------------------
    # each epoch
    # ----------------------------------------
    since = time.time()

    loss, acc1, acc5, class_acc = test(test_loader, model, criterion, device, args)

    print('-' * 50)
    print('COMPLETE !!!')
    print(class_acc)
    print(loss, acc1, acc5)
    print('TIME CONSUMED', time.time() - since)


def test(test_loader, model, criterion, device, args):
    loss_meter = AverageMeter('Loss', ':.4e')
    acc1_meter = AverageMeter('Acc@1', ':6.2f')
    acc5_meter = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(total=len(test_loader), step=20, prefix='Test',
                             meters=[loss_meter, acc1_meter, acc5_meter])
    class_acc = metrics.ClassAccuracy(args.num_classes)
    model.eval()

    for i, samples in enumerate(test_loader):
        inputs, labels, _ = samples
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            acc1, acc5 = metrics.accuracy(outputs, labels, topk=(1, 5))
            class_acc.accuracy(outputs, labels)

            loss_meter.update(loss.item(), inputs.size(0))
            acc1_meter.update(acc1.item(), inputs.size(0))
            acc5_meter.update(acc5.item(), inputs.size(0))

            progress.display(i)

    return loss_meter, acc1_meter, acc5_meter, class_acc


if __name__ == '__main__':
    main()
