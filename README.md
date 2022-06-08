## Model Doctor: A Simple Gradient Aggregation Strategy for Diagnosing andTreating CNN Classifiers

![](framework.png)

This is a PyTorch implementation of the [Model Doctor](https://arxiv.org/pdf/2112.04934.pdf):
```
@article{feng2021model,
  title={Model Doctor: A Simple Gradient Aggregation Strategy for Diagnosing and Treating CNN Classifiers},
  author={Feng, Zunlei and Hu, Jiacong and Wu, Sai and Yu, Xiaotian and Song, Jie and Song, Mingli},
  journal={arXiv preprint arXiv:2112.04934},
  year={2021}
}
```
### Environment
+ python 3.8

### Command
#### 1. Train a pre-trained model
```shell
bash scripts/train.sh
```
#### 2. Prepare for channel constraints
1. Sift high confidence images
 ```shell
 bash scripts/image_sift.sh
 ```
2. Calculate the gradient for each class
 ```shell
 bash scripts/grad_calcualte.sh
 ```

#### 3. Prepare for spatial constraints
1. Sift low confidence images
 ```shell
 bash scripts/image_sift.sh
 ```
2. Label the foreground area with [labelme](https://github.com/wkentaro/labelme)
3. Convert labelme annotation files into masks
```
python preprocessing/labelme_to_mask.py
```

#### 4. Train the model with the model doctor:
```shell
bash scripts/train_model_doctor.sh
```



