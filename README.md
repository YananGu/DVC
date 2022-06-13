# DVC
Code For CVPR2022 paper "Not Just Selection, but Exploration: Online Class-Incremental Continual Learning via Dual View Consistency"

## Usage

### Requirements
requirements.txt

### Data preparation
- CIFAR10 & CIFAR100 will be downloaded during the first run. (datasets/cifar10;/datasets/cifar100)
- Mini-ImageNet: Download from https://www.kaggle.com/whitemoon/miniimagenet/download, and place it in datasets/mini_imagenet/


### CIFAR-100
```shell
  python general_main.py --data  cifar100 --cl_type nc --agent ER_DVC  --retrieve MGI --update random --mem_size 1000 --dl_weight 4.0
 ```

 ### CIFAR-10
```shell
  python general_main.py --data cifar10 --cl_type nc --agent ER_DVC --retrieve MGI --update random --mem_size 200 --dl_weight 2.0  --num_tasks 5
 ```
 
 ### Mini-Imagenet
```shell
python general_main.py --data  mini_imagenet --cl_type nc --agent ER_DVC  --retrieve MGI --update random --mem_size 1000 --dl_weight 4.0
 ```
 
 
 
 
 
 ## Reference
[online-continual-learning](https://github.com/RaptorMai/online-continual-learning)
[agmax](https://github.com/roatienza/agmax)
 
