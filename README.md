# Plant Segmentation Experiments

This is a simple project to demonstrate how to use RFlow to create a
simple semantic segmentation experiment.

The workflow is show in the above image, and it's coded on the `workflow.py`. 

* The `data` graph
  - dataset
  - split dataset
* The `fpn_resnet` graphs
  - train
  - predict_single
  - 



```shell
$ rflow fpn_resnet run test
```


## Executing - Directly

Required/Tested system:

* Ubuntu>=18.04
* Python>=3.7

Create your virtual environment or conda environment, and the install the requirements:

```shell
pip3 install -r requirements.txt
```

## Executing - Using Docker





