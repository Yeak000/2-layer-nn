# 2-layer-nn
2-layer-nn with CuPy

train.py中包含
载入数据（load_mnist_data）

独热编码（one_hot）

激活函数（ReLu）

Softmax

参数初始化（initialize_parameters）

前向传播（forward_propagation）

loss（compute_loss包含了L2正则化在内）

梯度的计算及反向传播（backward_propagation）

学习率下降（learning_rate_decay）

优化器SGD（SGD）

计算准确率（compute_accuracy）

保存模型（save_model）

训练（train）

param_search.py中包含

参数搜索（param_search）

函数param_search的输入为三个分别代表学习率，隐藏层单元数，l2正则化的待搜索列表，运行过程中会同时保存模型到ckpts文件夹中

输出一个字典，其中key包含了学习率，隐藏层单元数，l2正则化的具体参数，value包含模型训练最后一轮在验证集上的loss

eval.py读取ckpts文件夹中保存的模型并在测试集上测试模型效果

另附kaggle notebook链接：https://www.kaggle.com/code/yeakguo/nn-hm1
