#!/bin/bash
export result_path='/nfs3-p1/hjc/md/output/'
export exp_name='vgg16_cifar10_202206072149'
export model_name='vgg16'
export data_name='cifar10'
export in_channels=3
export num_classes=10
export model_path=${result_path}${exp_name}'/models/model_ori.pth'
export data_path='/nfs3-p1/hjc/datasets/cifar10/test'
export device_index='2'
python test.py \
  --model_name ${model_name} \
  --data_name ${data_name} \
  --in_channels ${in_channels} \
  --num_classes ${num_classes} \
  --model_path ${model_path} \
  --data_path ${data_path} \
  --device_index ${device_index}
