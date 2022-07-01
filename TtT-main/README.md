# TtT
code for our ACL2021 paper "Tail-to-Tail Non-Autoregressive Sequence Prediction for Chinese Grammatical Error Correction"

The pretrained BERT model can be downloaded from: https://drive.google.com/file/d/1gX9YYcGpR44BsUgoJDtbWqV-4PsaWpq1/view?usp=sharing

Training:
```
./train.sh
```

Tips to reproduce the results:
- More epochs: more than 200;
- Larger batchsize on GPUs with large memory such as V100.

-------------------------------------------------------------------------------------
lyy：
主模型为main.py中的myModel。模型首先读入句子中的每个字符，进行变长处理，然后输入预训练的bert层进行编码，再通过tranformer层和和线性层映射到词表。此时，得到了每个句子的每个token对应词表中每个词的概率。接下来，将其输入CRF层，再取top-k，生成的正确句输出。
1.模型读入句子中的每个字符。对每个batch构建文本矩阵、对应的分类标签矩阵，mask矩阵，对于输入句和答案输出句不同的情况，首先将其进行变长处理。（data_loader.py）
2.将三者输入bert层进行编码，输出维度为[seq\_len, batch\_size, embedding\_size]。(bert.py)
3.将编码输出通过tranformer层(bert.py)
4.经过一个线性层和softmax映射到词表，词表共有16541个词，即模型的分类数，输出维度为[seq\_len, batch\_size, num\_class]。
5.将每个句子的16541维分类结果输入CRF层，使用$loss_{dp}$和$loss_{crf}$二者结合作为损失函数，输出[seq\_len, batch\_size, num\_class]维的分类结果(crf_layer.py)
6.取top-k的值映射到词表，作为生成的正确句输出

其最终的损失函数由两部分组成，loss_{dp}和loss_{crf}
1. loss_{dp}为在正确标签上输出的预测概率值取log，对batch内的每个句子求和，其中1-pdl的gamma次方为惩罚项，增加的错误token的比重，这也是论文的创新点之一
2. loss_{crf}与loss_{dp}相似，但概率值的计算不同。Pcrf的分子分为2个部分，一是发射矩阵的分数s，即已知序列的情况下，对应标签出现的概率，即与loss_{dp}相同，每个位置的字符在正确标签上输出的预测概率；二是标签转移矩阵t，即对于一个句子，已知上一个字符的标签，下一个对应标签出现的概率。由于词表数量大，这里采用低秩分解的方法，减少了模型的参数量。
Pcrf的分母是归一化值，全部可能出现的标签序列的概率之和，由于朴素算法时间复杂度较高，使用基于动态规划的Viterbi算法计算得到。

最终的loss函数为二者相加，由公式给出，增加惩罚项。