基于Pytorch的卷积神经网络MNIST手写数字识别 <br>
该README是对cnn_mnist_pytorch.py工程的说明

## 一、环境要求:
 - Windows10
 - PyCharm
 - conda4.8.2 
 - NVIDIA GPU(可选) 
	
	
 - python 3.7.6
 - pytorch 1.4.0
 - torchvision 0.5.0
 - numpy 1.18.1 
 - matplotlib 3.1.3
 - time 
 - cudatoolkit(可选) 10.1 

 - MNIST 

## 二、使用方法:
```
1.正确设置路径
├── cnn_mnist_pytorch.py   #主程序 
├── modelpara.pth          #已训练网络参数模型 
├── README.txt             #使用说明 
├── MNIST                  #MNIST数据集 
│   ├── processed
└── └── raw
```
```
2.直接运行cnn_mnist_pytorch.py即可获得已训练模型的测试结果
注:若无GPU，请将use_gpu设为0
```
```
3.设置超参数以重新训练
```

## 三、测试结果:
MNIST测试集识别准确率99.22%
10000张测试集图片识别总时间2.362s(GPU)/8.283s(CPU)

-------------- <br>
By: Mr.Liu Mr.Li Ms.Gai<br>
时间：2020年4月 <br>
