# DVC
Code For CVPR2022 paper "[Not Just Selection, but Exploration: Online Class-Incremental Continual Learning via Dual View Consistency](https://openaccess.thecvf.com/content/CVPR2022/papers/Gu_Not_Just_Selection_but_Exploration_Online_Class-Incremental_Continual_Learning_via_CVPR_2022_paper.pdf)"

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


If our code or models help your work, please cite our [paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Gu_Not_Just_Selection_but_Exploration_Online_Class-Incremental_Continual_Learning_via_CVPR_2022_paper.pdf):

```shell
@InProceedings{Gu_2022_CVPR,
    author    = {Gu, Yanan and Yang, Xu and Wei, Kun and Deng, Cheng},
    title     = {Not Just Selection, but Exploration: Online Class-Incremental Continual Learning via Dual View Consistency},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {7442-7451}
}
```
