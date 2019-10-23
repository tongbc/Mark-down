# NLP知识

### 自编码和自回归模型

![1564909673565](C:\Users\tbc\AppData\Roaming\Typora\typora-user-images\1564909673565.png)

如上所示分别为自回归模型与自编码模型，其中黄色块为输入字符，蓝色块为字符的位置。对于自回归语言模型，它希望通过已知的前半句预测后面的词或字。对于自编码语言模型，它希望通过一句话预测被 Mask 掉的字或词，如上所示第 2 个位置的词希望通过第 1、3、5 个词进行预测。

以前，最常见的语言模型就是自回归式的了，它的计算效率比较高，且明确地建模了概率密度。但是自回归语言模型有一个缺陷，它只能编码单向语义，不论是从左到右还是从右到左都只是单向语义。这对于下游 NLP 任务来说是致命的，因此也就有了 BERT 那种自编码语言模型。

BERT 通过预测被 Mask 掉的字或词，从而学习到双向语义信息。但这种任务又带来了新问题，它只是建模了近似的概率密度，因为 BERT 假设要预测的词之间是相互独立的，即 Mask 之间相互不影响。此外，自编码语言模型在预训练过程中会使用 MASK 符号，但在下游 NLP 任务中并不会使用，因此这也会造成一定的误差。

从而将上面两类模型的优点结合起来。XLNet 采用了一种新型语言建模任务，它通过随机排列自然语言而预测某个位置可能出现的词。如下图所示为排列语言模型的预测方式：

![](D:\md_images\xlnet2.jpg)

这两件事是有冲突的，如果模型需要预测位置 2 的「喜欢」，那么肯定不能用该位置的内容向量。但与此同时，位置 2 的完整向量还需要参与位置 5 的预测，且同时不能使用位置 5 的内容向量。

这类似于条件句：如果模型预测当前词，则只能使用位置向量；如果模型预测后续的词，那么使用位置加内容向量。因此这就像我们既需要标准 Transformer 提供内容向量，又要另一个网络提供对应的位置向量。

针对这种特性，研究者提出了 Two-Stream Self-Attention，它通过构建两条路径解决这个条件句。如下所示为 Two-Stream 的结构，其中左上角的 a 为 Content 流，左下角的 b 为 Query 流，右边的 c 为排列语言模型的整体建模过程。

![](D:\md_images\xlnet3.jpg)

在 Content 流中，它和标准的 Transformer 是一样的，第 1 个位置的隐藏向量 h_1 同时编码了内容与位置。在 Query 流中，第 1 个位置的隐向量 g_1 只编码了位置信息，但它同时还需要利用其它 Token 的内容隐向量 h_2、h_3 和 h_4，它们都通过 Content 流计算得出。因此，我们可以直观理解为，Query 流就是为了预测当前词，而 Content 流主要为 Query 流提供其它词的内容向量。

上图 c 展示了 XLNet 的完整计算过程，e 和 w 分别是初始化的词向量的 Query 向量。注意排列语言模型的分解顺序是 3、2、4、1，因此 Content 流的 Mask 第一行全都是红色、第二行中间两个是红色，这表明 h_1 需要用所有词信息、h_2 需要用第 2 和 3 个词的信息。此外，Query 流的对角线都是空的，表示它们不能使用自己的内容向量 h。

### 矩阵的「秩」

- 「秩」是图像经过矩阵变换之后的空间维度

  「秩」是列空间的维度

- 



### PCA  LDA

- PCA的思想是将n维特征映射到k维上（k<n），这k维是全新的正交特征。这k维特征称为主成分，是重新构造出来的k维特征，而不是简单地从n维特征中去除其余n-k维特征。

[PCA LDA博客](https://blog.csdn.net/kuweicai/article/details/79255270)

![协方差矩阵](D:\md_images\协方差矩阵.jpg)

对角线上分别是x和y的方差，非对角线上是协方差。协方差是衡量两个变量同时变化的变化程度。协方差大于0表示x和y若一个增，另一个也增；小于0表示一个增，一个减。如果ｘ和ｙ是统计独立的，那么二者之间的协方差就是０；但是协方差是０，并不能说明ｘ和ｙ是独立的。协方差绝对值越大，两者对彼此的影响越大，反之越小。协方差是没有单位的量，因此，如果同样的两个变量所采用的量纲发生变化，它们的协方差也会产生数值上的变化。

第三步，求协方差的特征值和特征向量，得到

![](https://img-blog.csdn.net/20150304201031902)

上面是两个特征值，下面是对应的特征向量，特征值0.0490833989对应特征向量为(-0.735178656, 0.677873399)，这里的特征向量都归一化为单位向量。

第四步，将特征值按照从大到小的顺序排序，选择其中最大的k个，然后将其对应的k个特征向量分别作为列向量组成特征向量矩阵。

这里特征值只有两个，我们选择其中最大的那个，这里是1.28402771，对应的特征向量是(-0.677873399, -0.735178656)T。

第五步，将样本点投影到选取的特征向量上。假设样例数为m，特征数为n，减去均值后的样本矩阵为DataAdjust(m*n)，协方差矩阵是n*n，选取的k个特征向量组成的矩阵为EigenVectors(n*k)。那么投影后的数据FinalData为

FinalData(10*1) = DataAdjust(10*2矩阵) x 特征向量(-0.677873399, -0.735178656)T

得到结果为：

![](https://img-blog.csdn.net/20150304201345746)
--------------------- 

### Micro-F1和Macro-F1

最后看Micro-F1和Macro-F1。在第一个多标签分类任务中，可以对每个“类”，计算F1，显然我们需要把所有类的F1合并起来考虑。

这里有两种合并方式：

第一种计算出所有类别总的Precision和Recall，然后计算F1。

例如依照最上面的表格来计算:Precison=5/(5+4)=0.556,Recall=5/(5+4)=0.556，然后带入F1的公式求出F1，这种方式被称为Micro-F1微平均。

第二种方式是计算出每一个类的Precison和Recall后计算F1，最后将F1平均。

例如上式A类：P=2/(2+0)=1.0，R=2/(2+2)=0.5，F1=(2*1*0.5)/1+0.5=0.667。同理求出B类C类的F1，最后求平均值，这种范式叫做Macro-F1宏平均。

###  F1,precision,recall

[全部的]https://zhuanlan.zhihu.com/p/34079183


某池塘有1400条鲤鱼，300只虾，300只鳖。现在以捕鲤鱼为目的。撒一大网，逮着了700条鲤鱼，200只虾，100只鳖。那么，这些指标分别如下：

精确率 = 700 / (700 +200 + 100) = 70%
召回率 = 700 / 1400 =50%

 

可以吧上述的例子看成分类预测问题，对于“鲤鱼来说”，TP真阳性为700，FP假阳性为300，FN假阴性为700。

Precison=TP/(TP+FP)=700(700+300)=70%
Recall=TP/(TP+FN)=700/(700+700)=50%

 

将上述例子，改变一下：把池子里的所有的鲤鱼、虾和鳖都一网打尽，观察这些指标的变化。

精确率 = 1400 / (1400 +300 + 300) = 70%
召回率 = 1400 / 1400 =100%

 

TP为1400：有1400条鲤鱼被预测出来；FP为600：有600个生物不是鲤鱼类，却被归类到鲤鱼；FN为0，鲤鱼都被归类到鲤鱼类去了，并没有归到其他类。

Precision=TP/(TP+FP)=1400/(1400+600)=70%
Recall=TP/(TP+FN)=1400/(1400)=100%

### Attention

[苏建林老师 Attention解读](https://kexue.fm/archives/4765)

O_seq = Attention(8,16)([embeddings,embeddings,embeddings]) 这行代码里的8和16表达什么意思

Multi-Head Attention中的h和d~v（请通读本文并对照源码）

[muti-haed Attention苏神代码详解](https://zhuanlan.zhihu.com/p/67836133)

[遍地开花Attention](https://mp.weixin.qq.com/s/MzHmvbwxFCaFjmMkjfjeSg)

[讲的很清楚的bert 和 trans]https://www.cnblogs.com/xlturing/p/10824400.html

### transformer作用过程

**Transformer 工作流程**

Transformer的工作流程就是上面介绍的每一个子流程的拼接

- 输入的词向量首先叠加上Positional Encoding，然后输入至Transformer内
- 每个Encoder Transformer会进行一次Multi-head self attention->Add & Normalize->FFN->Add & Normalize流程，然后将输出输入至下一个Encoder中
- 最后一个Encoder的输出将会作为memory保留
- 每个Decoder Transformer会进行一次Masked Multi-head self attention->Multi-head self attention->Add & Normalize->FFN->Add & Normalize流程，其中Multi-head self attention时的K、V至来自于Encoder的memory。根据任务要求输出需要的最后一层Embedding。
- Transformer的输出向量可以用来做各种下游任务

### 样本不均衡问题

1. 上采样/下采样

   下采样，对于一个不均衡的数据，让目标值(如0和1分类)中的样本数据量相同，且以数据量少的一方的样本数量为准。上采样就是以数据量多的一方的样本数量为标准，把样本数量较少的类的样本数量生成和样本数量多的一方相同，称为上采样。

   下采样

   获取数据时一般是从分类样本多的数据从随机抽取等数量的样本

2. 分类损失函数

   通常二分类使用交叉熵损失函数，但是在样本不均衡下，训练时损失函数会偏向样本多的一方，造成训练时损失函数很小，但是对样本较小的类别识别精度不高。

   解决办法之一就是给较少的类别加权，形成加权交叉熵(Weighted cross entropy loss)。今天看到两个方法将权值作为类别样本数量的函数，其中有一个很有意思就录在这里。

   ![1568552629287](C:\Users\tbc\AppData\Roaming\Typora\typora-user-images\1568552629287.png)

   上边说明的时，正负样本的权值和他们的对方数量成比例，举个例子，比如正样本有30，负样本有70，那么正样本的权w+=70/(30+70)=0.7，负样本的权就是w-=30/(30+70)=0.3，

   这样算下来的权值是归一的。这种方法比较直观，普通，应该是线性的。

   ![1568552662253](C:\Users\tbc\AppData\Roaming\Typora\typora-user-images\1568552662253.png)

   这个的权值直接就是该类别样本数的反比例函数，是非线性的，相比于上边的很有意思，提供了另一种思路。为了统一期间还是使用w+,w-表示这里的beta P和beta N，

   举个例子，比如正样本有30，负样本有70，那么正样本的权w+=(30+70)/30=3.33，负样本的权就是w-=(30+70)/70=1.42。

3. focal loss

   https://blog.csdn.net/c9Yv2cf9I06K2A9E/article/details/78920998

### CRF

LTSM and CRF

https://www.zhihu.com/question/35866596/answer/139485548

作者：Scofield





既然LSTM都OK了，为啥researchers搞一个LSTM+CRF的hybrid model? 

哈哈，因为a single LSTM预测出来的标注有问题啊！举个segmentation例子(BES; char level)，plain LSTM 会搞出这样的结果：

> **input**: "学习出一个模型，然后再预测出一条指定"
> **expected output**: 学/B 习/E 出/S 一/B 个/E 模/B 型/E ，/S 然/B 后/E 再/E 预/B 测/E ……
> **real output**: 学/B 习/E 出/S 一/B 个/B 模/B 型/E ，/S 然/B 后/B 再/E 预/B 测/E ……

看到不，用LSTM，整体的预测accuracy是不错indeed, 但是会出现上述的错误：在B之后再来一个B。这个错误在CRF中是不存在的，因为CRF的特征函数的存在就是为了对given序列观察学习各种特征（n-gram，窗口），这些特征就是在限定窗口size下的各种词之间的关系。然后一般都会学到这样的一条规律（特征）：B后面接E，不会出现E。这个限定特征会使得CRF的预测结果不出现上述例子的错误。当然了，CRF还能学到更多的限定特征，那越多越好啊！

https://zhuanlan.zhihu.com/p/44042528

LOSS_FUNCTION = P_real_path/(P_all)

![1569574124604](C:\Users\tbc\AppData\Roaming\Typora\typora-user-images\1569574124604.png)

### 朴素贝叶斯为何朴素

https://zhuanlan.zhihu.com/p/26262151

**p(不帅、性格不好、身高矮、不上进|嫁) = p(不帅|嫁)\*p(性格不好|嫁)\*p(身高矮|嫁)\*p(不上进|嫁)**

等等，为什么这个成立呢？学过概率论的同学可能有感觉了，这个等式成立的条件需要特征之间相互独立吧！

**对的！这也就是为什么朴素贝叶斯分类有朴素一词的来源，朴素贝叶斯算法是假设各个特征之间相互独立，那么这个等式就成立了！**

**为什么需要假设特征之间相互独立呢？**

1、我们这么想，假如没有这个假设，那么我们对右边这些概率的估计其实是不可做的，这么说，我们这个例子有4个特征，其中帅包括{帅，不帅}，性格包括{不好，好，爆好}，身高包括{高，矮，中}，上进包括{不上进，上进}，**那么四个特征的联合概率分布总共是4维空间，总个数为2\*3\*3\*2=36个。**

**24个，计算机扫描统计还可以，但是现实生活中，往往有非常多的特征，每一个特征的取值也是非常之多，那么通过统计来估计后面概率的值，变得几乎不可做，这也是为什么需要假设特征之间独立的原因。**

2、假如我们没有假设特征之间相互独立，那么我们统计的时候，就需要在整个特征空间中去找，比如统计p(不帅、性格不好、身高矮、不上进|嫁),

**我们就需要在嫁的条件下，去找四种特征全满足分别是不帅，性格不好，身高矮，不上进的人的个数，这样的话，由于数据的稀疏性，很容易统计到0的情况。 这样是不合适的。**

根据上面俩个原因，朴素贝叶斯法对条件概率分布做了条件独立性的假设，由于这是一个较强的假设，朴素贝叶斯也由此得名！这一假设使得朴素贝叶斯法变得简单，但有时会牺牲一定的分类准确率。

### NLLLoss  CrossEntropyLoss

https://blog.csdn.net/qq_22210253/article/details/85229988

NLLLoss需要先softmax，再log，再送入

而CrossEntropyLoss

### torch

1.torch实现正则化

torch.optim

optimizer = optim.Adam(model.parameters(),lr=learning_rate,weight_decay=0.01)

只是L2正则化

就整体而言，对比加入正则化和未加入正则化的模型，训练输出的loss和Accuracy信息，我们可以发现，加入正则化后，loss下降的速度会变慢，准确率Accuracy的上升速度会变慢，并且未加入正则化模型的loss和Accuracy的浮动比较大（或者方差比较大），而加入正则化的模型训练loss和Accuracy，表现的比较平滑。并且随着正则化的权重lambda越大，表现的更加平滑。这其实就是正则化的对模型的惩罚作用，通过正则化可以使得模型表现的更加平滑，即通过正则化可以有效解决模型过拟合的问题。

### 优化器

[momentum 动量法]https://blog.csdn.net/tsyccnh/article/details/76270707

[优化器各参数]https://blog.csdn.net/qq_34690929/article/details/79932416

### batch normalization

https://www.cnblogs.com/guoyaohua/p/8724433.html

本质思想：BN的基本思想其实相当直观：因为深层神经网络在做非线性变换前的**激活输入值**（就是那个x=WU+B，U是输入）**随着网络深度加深或者在训练过程中，其分布逐渐发生偏移或者变动，之所以训练收敛慢，一般是整体分布逐渐往非线性函数的取值区间的上下限两端靠近**（对于Sigmoid函数来说，意味着激活输入值WU+B是大的负值或正值），所以这**导致反向传播时低层神经网络的梯度消失**，这是训练深层神经网络收敛越来越慢的**本质原因**，**而BN就是通过一定的规范化手段，把每层神经网络任意神经元这个输入值的分布强行拉回到均值为0方差为1的标准正态分布**，其实就是把越来越偏的分布强制拉回比较标准的分布，这样使得激活输入值落在非线性函数对输入比较敏感的区域，这样输入的小变化就会导致损失函数较大的变化，意思是**这样让梯度变大，避免梯度消失问题产生，而且梯度变大意味着学习收敛速度快，能大大加快训练速度。**

　　THAT’S IT。其实一句话就是：**对于每个隐层神经元，把逐渐向非线性函数映射后向取值区间极限饱和区靠拢的输入分布强制拉回到均值为0方差为1的比较标准的正态分布，使得非线性变换函数的输入值落入对输入比较敏感的区域，以此避免梯度消失问题。**因为梯度一直都能保持比较大的状态，所以很明显对神经网络的参数调整效率比较高，就是变动大，就是说向损失函数最优值迈动的步子大，也就是说收敛地快。BN说到底就是这么个机制，方法很简单，道理很深刻。

### 反向面试

https://github.com/yifeikong/reverse-interview-zh

### 损失函数

1. MSE mean square error 

   二次函数仅具有全局最小值。由于没有局部最小值，所以我们永远不会陷入它。因此，可以保证梯度下降将收敛到全局最小值(如果它完全收敛)。

   MSE损失函数通过平方误差来惩罚模型犯的大错误。把一个比较大的数平方会使它变得更大。但有一点需要注意，这个属性使MSE成本函数对异常值的健壮性降低。**因此，如果我们的数据容易出现许多的异常值，则不应使用这个它。**

2. 绝对误差损失

   L1Loss，**与MSE相比，MAE成本对异常值更加健壮**。

3. Huber损失

   对于较小的误差，它是二次的，否则是线性的(对于其梯度也是如此)。Huber损失对于异常值比MSE更强

4. 交叉熵，熵，KL散度

   https://blog.csdn.net/Dby_freedom/article/details/83374650

   https://www.jianshu.com/p/ae3932eda8f2

5. BCEWithLogitsLoss

   二分交叉熵，省略了sigmoid这一步



### 处理缺失值

https://www.kaggle.com/rtatman/data-cleaning-challenge-handling-missing-values

### NLP校招

https://zhuanlan.zhihu.com/p/62902811

[古老。。汇总！！！非常重要]https://zhuanlan.zhihu.com/p/41975491

[非常重要的transformer]https://zhuanlan.zhihu.com/p/48508221

[面试问题]https://zhuanlan.zhihu.com/p/55643274

[word2vec]https://www.zhihu.com/question/44832436/answer/266068967

[transformer源码]https://blog.csdn.net/stupid_3/article/details/83184691

[deep learning 调参]https://www.zhihu.com/question/41631631/answer/776852832

[SVM面试会问的]https://zhuanlan.zhihu.com/p/76946313

[京东问题]https://zhuanlan.zhihu.com/p/40870443

[LSTM]https://zhuanlan.zhihu.com/p/79064602    https://zhuanlan.zhihu.com/p/39191116

![1570690851740](C:\Users\tbc\AppData\Roaming\Typora\typora-user-images\1570690851740.png)

[torch的坑]https://www.zhihu.com/question/67209417/answer/835804637

[**非常关键的机器学习面试点！！！**]https://zhuanlan.zhihu.com/p/77587367

[归一化和标准化]https://www.jianshu.com/p/95a8f035c86c   https://blog.csdn.net/u010947534/article/details/86632819     https://www.cnblogs.com/chaosimple/p/4153167.html

[推荐系统 文总建议]https://www.zhihu.com/question/342267611/answer/805956104

举一个简单的例子，在KNN中，我们需要计算待分类点与所有实例点的距离。假设每个实例点（instance）由n个features构成。如果我们选用的距离度量为欧式距离，如果数据预先没有经过归一化，那么那些绝对值大的features在欧式距离计算的时候起了决定性作用。

从经验上说，归一化是让不同维度之间的特征在数值上有一定比较性，可以大大提高分类器的准确性。

**平方损失函数的东西，具体形式可以写成 ![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac%7B1%7D%7B2%7D%5Csum_%7B0%7D%5E%7Bn%7D%7B%7D%28y_%7Bi%7D-F%28x_%7Bi%7D%29%29%5E%7B2%7D) ，熟悉其他算法的原理应该知道，这个损失函数主要针对回归类型的问题，分类则是用熵值类的损失函数。**

[GBDT]https://zhuanlan.zhihu.com/p/29765582

[分词]https://zhuanlan.zhihu.com/p/50444885 

N最短路径分词：在上图中, 边的起点为词的第一个字, 边的终点为词尾的下一个字. 边1表示"我"字单字成词, 边2表示"只是"可以作为一个单词.

每个边拥有一个权值, 表示该词出现的概率. 最简单的做法是采用词频作为权值, 也可以采用TF-IDF值作为权值提高对低频词的分词准确度.

N最短路径分词即在上述有向无环图中寻找N条权值和最大的路径, 路径上的边标志了最可能的分词结果.通常我们只寻找权值和最大的那一条路径.

[牛逼！！非常重要]http://ddrv.cn/a/146186

[word2vec]https://zhuanlan.zhihu.com/p/35074402  https://blog.csdn.net/itplus/article/details/37969979

hierarchical softmax：输出层变成霍夫曼树，更新每次二分的矩阵，cbow就将梯度更新回之前的每个词上

negative sampling：![1570271900204](C:\Users\tbc\AppData\Roaming\Typora\typora-user-images\1570271900204.png)

![1570271958107](C:\Users\tbc\AppData\Roaming\Typora\typora-user-images\1570271958107.png)

[labelEncoding,mean encoding]https://zhuanlan.zhihu.com/p/26308272

1. LabelEncoder编码高基数定性特征，虽然**只需要一列**，但是每个自然数都具有不同的重要意义，对于y而言**线性不可分**。使用简单模型，容易欠拟合（underfit），无法完全捕获不同类别之间的区别；使用复杂模型，容易在其他地方过拟合（overfit）。
2. OneHotEncoder编码高基数定性特征，必然产生**上万列的稀疏矩阵**，易消耗大量内存和训练时间，除非算法本身有相关优化（例：SVM）。

[FE by Chris]https://www.kaggle.com/c/ieee-fraud-detection/discussion/108575#latest-643395

[梯度消失]https://zhuanlan.zhihu.com/p/28687529

[LSTM解决梯度爆炸&消失]https://zhuanlan.zhihu.com/p/28749444

[机器学习经常问的问题]https://zhuanlan.zhihu.com/p/81253435

[分词]https://zhuanlan.zhihu.com/p/33261835

N-gram  正向最大，反向最大匹配，trie树   [结巴原理]https://www.cnblogs.com/echo-cheng/p/7967221.html

[维特比]https://www.zhihu.com/question/20136144/answer/239971177

[序列标注]https://blog.csdn.net/Jason__Liang/article/details/81772632

https://zhuanlan.zhihu.com/p/79552594		https://zhuanlan.zhihu.com/p/42096344