import os
import argparse
import time
from tqdm import tqdm

import torch
from torch import nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter

import loaders
import models
import metrics
from utils.train_util import AverageMeter, ProgressMeter
from core.grad_constraint import GradConstraint
from core.grad_noise import GradNoise


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model_name', default='', type=str, help='model name')
    parser.add_argument('--data_name', default='', type=str, help='data name')
    parser.add_argument('--in_channels', default='', type=int, help='in channels')
    parser.add_argument('--num_classes', default='', type=int, help='num classes')
    parser.add_argument('--num_epochs', default=200, type=int, help='num epochs')
    parser.add_argument('--ori_model_path', default='', type=str, help='original model path')
    parser.add_argument('--res_model_path', default='', type=str, help='result model path')
    parser.add_argument('--data_dir', default='', type=str, help='data dir')
    parser.add_argument('--log_dir', default='', type=str, help='log dir')
    parser.add_argument('--mask_dir', default=None, type=str, help='mask dir')
    parser.add_argument('--grad_dir', default='', type=str, help='grad dir')
    parser.add_argument('--alpha', default=0, type=float, help='weight coefficient for channel loss')
    parser.add_argument('--beta', default=0, type=float, help='weight coefficient for spatial loss')
    parser.add_argument('--device_index', default='0', type=str, help='device index')
    args = parser.parse_args()

    # ----------------------------------------
    # basic configuration
    # ----------------------------------------
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_index
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    train_path = os.path.join(args.data_dir, 'train')
    test_path = os.path.join(args.data_dir, 'test')
    mask_path = args.mask_dir  # for train set
    grad_path = os.path.join(args.grad_dir, 'layer_0.npy')

    print('-' * 50)
    print('TRAIN ON:', device)
    print('ORI PATH:', args.ori_model_path)
    print('RES PATH:', args.res_model_path)
    print('LOG DIR:', args.log_dir)
    print('-' * 50)

    # ----------------------------------------
    # trainer configuration
    # ----------------------------------------
    model = models.load_model(model_name=args.model_name, in_channels=args.in_channels, num_classes=args.num_classes)
    model.to(device)
    module = models.load_modules(model=model)[0]

    if args.beta == 0:
        train_loader = loaders.load_data(args.data_name, train_path, data_type='train')
        test_loader = loaders.load_data(args.data_name, test_path, data_type='test')
    else:  # load training set with mask
        train_loader = loaders.load_data_mask(args.data_name, train_path, mask_path, data_type='train')
        test_loader = loaders.load_data(args.data_name, test_path, data_type='test')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params=model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.num_epochs)

    writer = SummaryWriter(args.log_dir)

    # ----------------------------------------
    # model doctor configuration
    # ----------------------------------------
    constraint = GradConstraint(module=module, grad_path=grad_path, alpha=args.alpha, beta=args.beta)
    noise = GradNoise(module=module)

    # ----------------------------------------
    # each epoch
    # ----------------------------------------
    since = time.time()

    best_acc = None
    best_epoch = None

    for epoch in tqdm(range(args.num_epochs)):
        # noise.add_noise()
        loss_cls, loss_c, loss_s, acc1, acc5 = train(train_loader, model, criterion, constraint, optimizer, device)
        writer.add_scalar(tag='training loss cls', scalar_value=loss_cls.avg, global_step=epoch)
        writer.add_scalar(tag='training loss c', scalar_value=loss_c.avg, global_step=epoch)
        writer.add_scalar(tag='training loss s', scalar_value=loss_s.avg, global_step=epoch)
        writer.add_scalar(tag='training acc1', scalar_value=acc1.avg, global_step=epoch)
        # noise.remove_noise()
        loss_cls, loss_c, loss_s, acc1, acc5 = test(test_loader, model, criterion, constraint, device)
        writer.add_scalar(tag='test loss cls', scalar_value=loss_cls.avg, global_step=epoch)
        writer.add_scalar(tag='test loss c', scalar_value=loss_c.avg, global_step=epoch)
        writer.add_scalar(tag='test loss s', scalar_value=loss_s.avg, global_step=epoch)
        writer.add_scalar(tag='test acc1', scalar_value=acc1.avg, global_step=epoch)

        # ----------------------------------------
        # save best model
        # ----------------------------------------
        if best_acc is None or best_acc < acc1.avg:
            best_acc = acc1.avg
            best_epoch = epoch
            torch.save(model.state_dict(), args.res_model_path)

        scheduler.step()

    print('COMPLETE !!!')
    print('BEST ACC', best_acc)
    print('BEST EPOCH', best_epoch)
    print('TIME CONSUMED', time.time() - since)


def train(train_loader, model, criterion, constraint, optimizer, device):
    loss_cls_meter = AverageMeter('Loss CLS', ':.4e')
    loss_c_meter = AverageMeter('Loss C', ':.4e')  # channel
    loss_s_meter = AverageMeter('Loss S', ':.4e')  # spatial
    acc1_meter = AverageMeter('Acc@1', ':6.2f')
    acc5_meter = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(total=len(train_loader), step=20, prefix='Training',
                             meters=[loss_cls_meter, loss_c_meter, loss_s_meter, acc1_meter, acc5_meter])
    model.train()

    for i, samples in enumerate(train_loader):
        inputs, labels, xxx = samples
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss_cls = criterion(outputs, labels)
        loss_c = constraint.loss_channel(outputs, labels)
        loss_s = constraint.loss_spatial(outputs, labels, xxx)
        acc1, acc5 = metrics.accuracy(outputs, labels, topk=(1, 5))

        loss_cls_meter.update(loss_cls.item(), inputs.size(0))
        loss_c_meter.update(loss_c.item(), inputs.size(0))
        loss_s_meter.update(loss_s.item(), inputs.size(0))
        acc1_meter.update(acc1.item(), inputs.size(0))
        acc5_meter.update(acc5.item(), inputs.size(0))

        optimizer.zero_grad()  # 1
        loss = loss_cls + loss_c + loss_s
        loss.backward()  # 2
        optimizer.step()  # 3

        progress.display(i)

    return loss_cls_meter, loss_c_meter, loss_s_meter, acc1_meter, acc5_meter


def test(test_loader, model, criterion, constraint, device):
    loss_cls_meter = AverageMeter('Loss CLS', ':.4e')
    loss_c_meter = AverageMeter('Loss C', ':.4e')  # channel
    loss_s_meter = AverageMeter('Loss S', ':.4e')  # spatial
    acc1_meter = AverageMeter('Acc@1', ':6.2f')
    acc5_meter = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(total=len(test_loader), step=20, prefix='Training',
                             meters=[loss_cls_meter, loss_c_meter, loss_s_meter, acc1_meter, acc5_meter])
    model.eval()

    for i, samples in enumerate(test_loader):
        inputs, labels, xxx = samples
        inputs = inputs.to(device)
        labels = labels.to(device)

        # with torch.set_grad_enabled(False):
        outputs = model(inputs)
        loss_cls = criterion(outputs, labels)
        loss_c = constraint.loss_channel(outputs, labels)
        loss_s = constraint.loss_spatial(outputs, labels, xxx)
        acc1, acc5 = metrics.accuracy(outputs, labels, topk=(1, 5))

        loss_cls_meter.update(loss_cls.item(), inputs.size(0))
        loss_c_meter.update(loss_c.item(), inputs.size(0))
        loss_s_meter.update(loss_s.item(), inputs.size(0))
        acc1_meter.update(acc1.item(), inputs.size(0))
        acc5_meter.update(acc5.item(), inputs.size(0))

        progress.display(i)

    return loss_cls_meter, loss_c_meter, loss_s_meter, acc1_meter, acc5_meter


if __name__ == '__main__':
    main()
