# DVC
Code For CVPR2022 paper "Not Just Selection, but Exploration: Online Class-Incremental Continual Learning via Dual View Consistency"

## Usage

### CIFAR-100
```shell
  python general_main.py --data  cifar100 --cl_type nc --agent ER_DVC  --retrieve MGI --update random --mem_size 1000 --dl_weight 4.0
 ```

 ### CIFAR-10
```shell
  python general_main.py --data cifar10 --cl_type nc --agent ER_DVC --retrieve MGI --update random --mem_size 200 --dl_weight 2.0  --num_tasks 5
 ```
 
 ### mini-Imagenet
```shell
python general_main.py --data  mini_imagenet --cl_type nc --agent ER_DVC  --retrieve MGI --update random --mem_size 5000 --dl_weight 4.0
 ```
 
 
 
 
 
 
 Our code is support by [online-continual-learning](https://github.com/RaptorMai/online-continual-learning)
 
