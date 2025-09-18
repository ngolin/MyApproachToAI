# My Approach to AI

## 一、算法原理篇

### 1. [神经网络与深度学习 >>](https://github.com/ngolin/NNxDL)

人工智能的强大能力建立在一系列严谨的数学原理和算法之上，理解这些原理和算法是解开人工智能黑盒的关键。首先，设计一个神经网络模型，其中多参数的层级结构和非线性的激活函数为求解问题提供无限可能，然而手动调节这些参数找出这个可能并不可行；更可行的手段是通过定义一个损失函数，在已知的数据上计算损失值，再利用这个损失值自动更新和训练这些参数，使得损失值越来越小，这个过程就称为学习，这种学习方法就称为梯度下降，反向传播是梯度下降在神经网络中的具体实现算法，而优化器确保学习过程又快又好，正则化确保学习效果稳健可靠。

### 2. [Transformer 架构 >>](https://github.com/ngolin/AttentionIsAllYouNeed)

Transformer 模型架构作为大语言模型（LLM）的基石，起源于 [Attention Is All You Need](https://arxiv.org/abs/1706.03762) 这篇论文。随着 LLM 的大热，吸引了很多研究者复现这篇论文。我参考 [PyTorch 官方实现](https://github.com/pytorch/pytorch/blob/0d9c95cd7ee299e2e8c09df26d395be8775b506b/torch/nn/modules/transformer.py#L57)等资料，提供了一个简洁的实现。通过深入细致的实现，我感觉无论多么精妙的模型设计都离不开 FNN 的底色。当我们遇到新的问题或面临全新领域时，或许也要回到 FNN 的基础上进行求解和创新，通过拆解其层级结构，调整其连接方式，使 FNN 更有效地表征和适配数据；通过权重矩阵的多信号感知和激活函数的非线性变换，使 FNN 更高效地聚合和传递信息，从而赋予新的机制，解决新的问题。

## 二、工具使用篇

### 3. 熟悉 ndarray 核心计算层

### 4. 熟悉 torch.nn 神经网络层

### 5. 熟悉 Matplotlib 画图工具

## 三、项目实践篇

### 6. [NLP from Scratch >>](https://github.com/ngolin/nil2nlp)

### 7. [Seq2Seq >>](https://github.com/ngolin/Seq2seq/)
