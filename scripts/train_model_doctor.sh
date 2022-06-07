#!/bin/bash
export result_path='/nfs3-p1/hjc/md/output/'
export exp_name='vgg16_cifar10_202206072149'
export model_name='vgg16'
export data_name='cifar10'
export in_channels=3
export num_classes=10
export num_epochs=200
export ori_model_path=${result_path}${exp_name}'/models/model_ori.pth'
export res_model_path=${result_path}${exp_name}'/models/model_optim.pth'
export data_dir='/nfs3-p1/hjc/datasets/cifar10'
export log_dir=${result_path}'/runs/'${exp_name}
export mask_dir=${result_path}${exp_name}'/masks'
export grad_dir=${result_path}${exp_name}'/grads_50'
export alpha=10.0
export beta=0.0
export device_index='2'
python train_model_doctor.py \
  --model_name ${model_name} \
  --data_name ${data_name} \
  --in_channels ${in_channels} \
  --num_classes ${num_classes} \
  --num_epochs ${num_epochs} \
  --ori_model_path ${ori_model_path} \
  --res_model_path ${res_model_path} \
  --data_dir ${data_dir} \
  --log_dir ${log_dir} \
  --mask_dir ${mask_dir} \
  --grad_dir ${grad_dir} \
  --alpha ${alpha} \
  --beta ${beta} \
  --device_index ${device_index}
