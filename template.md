

# 飞桨适配 nueralop

> RFC 文档相关记录信息

|              |                    |
| ------------ | -----------------  |
| 提交作者      |         Kai Qi     |
| 提交时间      |       2024-10-25   |
| RFC 版本号    | v1.0               |
| 依赖飞桨版本   | 3.0-beta                |
| 文件名        | 20241025_neuraloperator.md |

## 1. 概述

### 1.1 相关背景

> [飞桨科学计算工具组件开发大赛](https://github.com/PaddlePaddle/PaddleScience/issues/1000)


`neuraloperator`是一个用于在 PyTorch 中学习神经运算符的综合库。它是 Fourier Neural Operators 和 Tensorized Neural Operators 的官方实现。并且`neuraloperator`支持在函数空间之间学习映射。

为`neuraloperator`工具组件支持飞桨后端，可以提高飞桨用户开发科学计算模型的开发效率。

### 1.2 功能目标


1. 整理 `neuraloperator` 的所有公开 API；
2. 使用 paddle 的 python API 等价组合实现上述公开 API 的功能；
3. 参考 PyTorch 后端已有代码，撰写飞桨后端的单测文件，并自测通过。

### 1.3 意义


为`neuraloperator`支持飞桨后端，从而提高飞桨用户开发科学计算模型的开发效率。

## 2. PaddleScience 现状


当前的PaddleScience有`neuraloperator`相关的[神经算子](https://github.com/PaddlePaddle/PaddleScience/tree/develop/examples/neuraloperator)实现。

但是为了确保代码的合入和比赛结束后成果的可维护性，本方案将不采用 PaddleScience 中现有的 `neuraloperator` 实现，而是基于 `neuraloperator` 的源码，独立编写支持飞桨后端的文件。

## 3. 目标调研


参考的源代码为：[https://github.com/neuraloperator/neuraloperator/tree/0.3.0](https://github.com/neuraloperator/neuraloperator/tree/0.3.0)

`neuraloperator`源代码实现的8个模型分别为, `FNO1d`, `FNO2d`, `FNO3d`, `SFNO`, `UNO`, `TFNO1d`, `TFNO2d`, `TFNO3d`。

为`neuraloperator`支持飞桨后端，主要的难点在于：

（1）傅里叶卷积层中复数权重的转换。

（2）`TFNO`中使用tltorch进行张量分解，需要利用tensorly的API组合实现张量分解的功能。
## 4. 设计思路与实现方案


比赛要求：所有文件组织结构必须与原有代码保持一致（新增文件除外），原有的注释、换行、空格、开源协议等内容不能随意变动（新增内容除外），否则会严重影响代码合入和比赛结束后成果代码的维护。


所以本项目按照[`neuraloperator`](https://github.com/neuraloperator/neuraloperator/tree/0.3.0)的源码来撰写支持飞桨后端的文件，并且严格保持文件结构、名称、注释、空格等都相同。



## 5. 测试和验收的考量


所有测试结果均在 NVIDIA RTX 4090 GPU 上进行，操作系统为 Ubuntu 20.04，。测试分为三个独立的环境：环境1 使用 PyTorch 1.10（CUDA 12.3），环境2 使用 Paddle 3.0-beta（CUDA 12.3），环境1和环境2主要用于代码测试；环境3 包括 PyTorch 2.5.0（CUDA 11.8） 和 Paddle 3.0-beta（CUDA 11.8），用于模型初始化权重参数的转换。

### 5.1 模型前向对齐

#### 5.1.1 权重转化

为了保证模型前向对齐不受到模型参数不一致的影响，本项目对模型采用相同的权重参数进行初始化。生成相同权重参数的流程主要包括以下三个步骤：

&nbsp;&nbsp;（1）随机初始化neuraloperator-pytorch的官方模型参数并保存成 pytorch_model.pth；

&nbsp;&nbsp;（2）使用Paddle/models/tutorials/mobilenetv3_prod/Step1-5中的 torch2paddle.py 将 pytorch_model.pth 转化为 paddle_model.pdparams；

&nbsp;&nbsp;（3）将生成的 paddle_model.pdparams 加载到 neuraloperator-paddle模型中。


在模型转换过程中，PyTorch 和 Paddle 之间的一些参数需要特别处理，
尤其是傅里叶卷积层中的复数权重。由于这类权重在两个框架中的表示方式不同，
转换时需要将复数权重的实部和虚部分别提取、保存，并在加载时按相同方式进行处理，以确保模型参数的一致性。

#### 5.1.2 模型前向对齐验证（以FNO2d为例）

模型前向对齐验证主要分为以下四个步骤：

&nbsp;&nbsp;（1）将数据集如darcy_train_16.pt、darcy_test_16.pt、darcy_test_32.pt等转化为ndarray格式；（[https://github.com/neuraloperator/neuraloperator/tree/0.3.0/neuralop/datasets/data](https://github.com/neuraloperator/neuraloperator/tree/0.3.0/neuralop/datasets/datad)）

&nbsp;&nbsp;（2）在 Paddle 和 PyTorch 中分别对模型的 DataLoader 和 Datasets 进行对齐处理，确保两者一致；

&nbsp;&nbsp;（3）PyTorch 前向传播：定义 PyTorch 模型，加载权重，固定随机种子，基于 numpy 生成随机数，并转换为 `torch.Tensor`，送入网络，得到输出`y_pytorch`；

&nbsp;&nbsp;（4）飞桨前向传播：定义飞桨模型，加载步骤 5.1.1 中转换后的权重，将步骤（3）中的生成的随机数，转换为 `paddle.Tensor`，送入网络，获取输出`y_paddle`。

最终定义`y_diff=|y_pytorch-y_paddle|`。最终不同模型的结果如下，其中参数`layers`表示Fourier Layers的数目。

| 模型   | Max(y_diff)  | Min(y_diff)   | MSE(y_pytorch, y_paddle)   |备注|
|:-----:|:-----:|:-----:|:-----:|:-----:|
| FNO1d | 2.78e-05 | 1.11e-08  | 9.45e-11 |单精度训练|
| FNO2d | 2.69e-05 | 0.00 | 3.44e-11 |单精度训练|
| FNO3d | 4.75e-05 |0.00   | 5.66e-11  |单精度训练|
| UFNO(layers=1) | 1.03e-09 | 0.00 | 6.91e-21 |双精度训练|
| UFNO(layers=5) | 4.02e-02 |2.24e-05  | 2.99e-04  |双精度训练，具体见6.1|
| SFNO | 5.60e-02 | 7.88e-07  | 1.92e-04 |双精度训练|
| TFNO1d | &nbsp; |&nbsp;   | &nbsp;  |撰写中|
| TFNO2d | &nbsp; | &nbsp;  | &nbsp; |撰写中|
|TFNO3d | &nbsp; |&nbsp;   | &nbsp;  |撰写中|




### 5.2 模型训练对齐（以FNO2d为例）


FNO2d-torch的每个epoch训练loss（L2 norm）为`loss_pytorch`，学习率为`lr_pytorch`。FNO2d-paddle的训练loss（L2 norm）为`loss_paddle`. 学习率为`lr_paddle`。定义指标`loss_diff=|loss_pytorch-loss_paddle|`，`lr_diff=|lr_pytorch-lr_paddle|`。

FNO2d-torch与FNO2d-paddle使用数据集 darcy_train_16.npy并且设置`ntrains=32 `。

FNO2d-torch的优化器参数为：
```python
optimizer = torch.optim.Adam(model.parameters(), 
                             lr=8e-3, 
                             weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

```

FNO2d-paddle的优化器参数为：
```python
scheduler = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=0.008, T_max=30)
optimizer = paddle.optimizer.Adam(parameters=model.parameters(), 
                                  learning_rate=scheduler, 
                                  weight_decay=0.0001)
```


FNO2d-torch与FNO2d-paddle的训练loss和学习率对比结果如下：


| epoch   | loss_pytorch   | loss_paddle   |loss_diff|lr_pytorch|lr_paddle|lr_diff|
|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| 0 | 32.06811523 | 32.06811523  |0.00|0.00797809|0.00797809|0.00|
| 1 | 31.38529968 | 31.38491058  |3.89e-04|0.00791259|0.00791259|0.00|
| 2 | 27.89798164 | 27.89617157 |1.81e-03|0.00780423|0.00780423|0.00|
| 3| 30.61170769 | 30.61891174 |7.20e-03|0.00765418|0.00765418|0.00|
| 4| 21.70384979 | 21.70200921 |1.84e-03|0.0074641|0.0074641|0.00|
| 5| 21.37845802| 21.37961197|1.15e-03|0.00723607|0.00723607|0.00|
| 6| 20.06527138| 20.06911851|3.85e-03|0.00697258|0.00697258|0.00|
| 7| 16.89506531 | 16.89888763 |3.82e-03|0.00667652|0.00667652|0.00|
| 8| 15.51529121 | 15.51611137 |8.20e-04|0.00635114|0.00635114|0.00|
| 9| 14.76096916 | 14.75966835 |1.30e-03|0.00600000|0.00600000|0.00|

### 5.3 模型训练表现评估

#### 5.3.1 不同模型在训练时使用的数据
| 模型   | 数据集名称 |方程类型 |数据维度 |
|:-----:|:-----:|:-----:|:-----:|
| FNO1d |  burgers_lowres.mat |Burgers Equation |[32,3,16]|
| FNO2d | darcy_train_16.npy |Darcy Flow| [32,3,16,16]|
| FNO3d | NavierStokes_V1e-5_N1200_T20.mat |Navier-Stokes Equation |[32,1,64,64,10]|
| UFNO | darcy_train_16.npy |Darcy Flow| [32,3,16,16]|
| SFNO |  darcy_train_16.npy |Darcy Flow| [32,3,16,16]|
| TFNO1d | burgers_lowres.mat |Burgers Equation   |[32,3,16]  |
| TFNO2d |darcy_train_16.npy | Darcy Flow  | [32,3,16,16]|
| TFNO3d | NavierStokes_V1e-5_N1200_T20.mat |Navier-Stokes Equation   | [32,1,64,64,10] |

#### 5.3.2 不同模型训练loss的差异
| 模型   | Max(loss_diff)  | Min(loss_diff)   | MSE(loss_pytorch, loss_paddle)   | lr_diff   | 备注|
|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| FNO1d | 1.72e-02 | 3.05e-05  | 6.12e-05 |0.00 |单精度训练|
| FNO2d | 7.20e-03 | 0.00 | 9.18e-06 |0.00 |单精度训练|
| FNO3d | 7.50e-03 |1.14e-05   | 6.96e-06  |0.00 |单精度训练|
| UFNO(layers=1) | 1.10e-06 | 1.53e-08  | 4.62e-13 |0.00 |双精度训练|
| UFNO(layers=5) | 1.49e+01 |2.96e-02   | 2.84e+01  |0.00|双精度训练，具体见6.1|
| SFNO | 6.85e-00 | 7.93e-02  | 1.55e+01 |0.00 |双精度训练|
| TFNO1d | &nbsp; |&nbsp;   | &nbsp;  ||撰写中|
| TFNO2d | &nbsp; | &nbsp;  | &nbsp; ||撰写中|
|TFNO3d | &nbsp; |&nbsp;   | &nbsp;  ||撰写中|


#### 5.3.3 不同模型训练耗时 
使用 `timeit` 模块中的 `default_timer` 进行计时，各个模型在训练 10个epochs 时的耗时为 `10_epochs_time`，数据读取耗时为 `data_reader_time`，其中包括所有的数据预处理步骤（例如对 y 进行UnitGaussianNormalizer）,各个模型的耗时（s）如下：

| 模型   | 10_epochs_time_torch | data_reader_time_torch|10_epochs_time_paddle|data_reader_time_paddle| 
|:-----:|:-----:|:-----:|:-----:|:-----:|
| FNO1d | 1.47e-01 | 4.81e-03 |4.19e-01|1.77e-01|
| FNO2d | 1.74e-01 | 7.17e-03 |4.56e-01|1.84e-01| 
| FNO3d | 1.44 |1.57e-02   |1.42|2.54e-01| 
| UFNO(layers=1) | 6.50e-02 | 6.87e-03  |1.91e-01|1.80e-01| 
| UFNO(layers=5) | 1.89e-01 |6.87e-03  |3.88e-01|1.54e-01| 
| SFNO | 1.71e-01| 6.98e-03  |3.24e-01|2.06e-01| 
| TFNO1d | &nbsp;   |&nbsp;   | &nbsp;  |撰写中|
| TFNO2d | &nbsp;   | &nbsp;  | &nbsp; |撰写中|
| TFNO3d | &nbsp;   |&nbsp;   | &nbsp;  |撰写中|


## 6. 待解决的问题

## 6.1 UFNO(layers=5)

在训练模型`UFNO(layers=5)`时，Pytorch版本和Paddle版本的训练损失在epoch=2时出现了显著差异，具体表现如下

| epoch   | loss_pytorch   | loss_paddle   |loss_diff|lr_pytorch|lr_paddle|lr_diff|
|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| 0 | 31.99346241 | 32.02302024  |2.96e-02|0.00797809|0.00797809|0.00|
| 1 | 28.79406888 | 28.28792398  |5.06e-01|0.00791259|0.00791259|0.00|
| 2 | 47.18949491 | 62.09492780 |1.49e+01|0.00780423|0.00780423|0.00|
| 3| 32.03622775 | 25.50253190 |6.53e+00|0.00765418|0.00765418|0.00|
| 4| 27.85857869 | 28.94802254 |1.09e+00|0.0074641|0.0074641|0.00|
| 5| 27.42604250| 29.43171277 |2.01e+00|0.00723607|0.00723607|0.00|
| 6| 30.13455886| 28.78272349|1.35e+00|0.00697258|0.00697258|0.00|
| 7| 25.39807468 | 26.31120726 |9.13e-01|0.00667652|0.00667652|0.00|
| 8| 24.83115153 | 22.55190781 |2.28e+00|0.00635114|0.00635114|0.00|
| 9| 22.83329250 | 20.45876442 |2.37e+00|0.00600000|0.00600000|0.00|

由于模型`UFNO(layers=1)`时的训练结果是对齐的，因此初步判断模型UFNO（layers=5）时出现的问题是由误差累积引起的。目前该问题尚未解决。

## 6.2 SFNO

在训练模型`SFNO`时，Pytorch版本和Paddle版本的训练损失在epoch=3时出现了显著差异，具体表现如下


| epoch   | loss_pytorch   | loss_paddle   |loss_diff|lr_pytorch|lr_paddle|lr_diff|
|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| 0 | 32.06262686 | 32.14197592  |7.93e-02|0.00797809|0.00797809|0.00|
| 1 | 31.48523259 | 30.92742936  |5.58e-01|0.00791259|0.00791259|0.00|
| 2 | 27.51574327 | 24.31375748 |3.20e+00|0.00780423|0.00780423|0.00|
| 3| 28.54495172 | 35.39358056 |6.85e+00|0.00765418|0.00765418|0.00|
| 4| 22.02489485 | 17.31783389 |4.71e+00|0.0074641|0.0074641|0.00|
| 5| 22.83375406| 20.14916569 |2.68e+00|0.00723607|0.00723607|0.00|
| 6| 22.05632114| 19.80994133|2.25e+00|0.00697258|0.00697258|0.00|
| 7| 21.20442744 | 17.54419624 |3.66e+00|0.00667652|0.00667652|0.00|
| 8| 19.56874356 | 14.98493694 |4.58e+00|0.00635114|0.00635114|0.00|
| 9| 18.70933591 | 13.35811956 |5.35e+00|0.00600000|0.00600000|0.00|



## 6.3 TFNO

`TFNO`中使用了tltorch进行张量分解，tltorch是tensorly的torch版。因为Paddle已经接入了tenosorly的后端，后期尝试张量分解时能不能直接用tensorly里的API。
## 7. 可行性分析和排期规划


| 里程碑        |  时间点     |
| -------------| ------------ | 
| 提交RFC      |     2024.10.25        |    
| 提交PR，修改代码完成合入          |    2024.11.25       | 
## 8. 影响面


为 PaddleScience 添加`neuraloperator`库。
