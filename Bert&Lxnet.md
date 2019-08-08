# Bert && XLNet

##  seq2seq

![1563426848570](C:\Users\tbc\AppData\Roaming\Typora\typora-user-images\1563426848570.png)

Seq2Seq其实就是**Encoder-Decoder**结构的网络，它的输入是一个序列，输出也是一个序列。在Encoder中，将序列转换成一个固定长度的向量，然后通过Decoder将该向量转换成我们想要的序列输出出来。

Encoder和Decoder一般都是RNN，通常为LSTM或者GRU.

在Encoder中，“欢迎/来/北京”这些词转换成词向量，也就是Embedding，我们用 ![[公式]](https://www.zhihu.com/equation?tex=v_%7Bi%7D) 来表示，与上一时刻的隐状态 ![[公式]](https://www.zhihu.com/equation?tex=h_%7Bi-1%7D) 按照时间顺序进行输入，每一个时刻输出一个隐状态 ![[公式]](https://www.zhihu.com/equation?tex=h_%7Bi%7D) ，我们可以用函数 ![[公式]](https://www.zhihu.com/equation?tex=f) 表达RNN隐藏层的变换： ![[公式]](https://www.zhihu.com/equation?tex=h_%7Bi%7D%3Df_%7B%7D%28v_%7Bi%7D%2Ch_%7Bi-1%7D%29) 。假设有t个词，最终通过Encoder自定义函数 ![[公式]](https://www.zhihu.com/equation?tex=q) 将各时刻的隐状态变换为向量 ![[公式]](https://www.zhihu.com/equation?tex=c) ： ![[公式]](https://www.zhihu.com/equation?tex=c%3Dq%28h_%7B0%7D%2C...%2Ch_%7Bt%7D%29) ，这个 ![[公式]](https://www.zhihu.com/equation?tex=c_%7B%7D) 就相当于从“欢迎/来/北京”这几个单词中提炼出来的大概意思一样，包含了这句话的含义。

Decoder的每一时刻的输入为Eecoder输出的 ![[公式]](https://www.zhihu.com/equation?tex=c_%7B%7D) 和Decoder前一时刻解码的输出 ![[公式]](https://www.zhihu.com/equation?tex=s_%7Bi-1%7D) *，*还有前一时刻预测的词的向量 ![[公式]](https://www.zhihu.com/equation?tex=E_%7Bi-1%7D) （如果是预测第一个词的话，此时输入的词向量为“_GO”的词向量，标志着解码的开始），我们可以用函数  

![[公式]](https://www.zhihu.com/equation?tex=s_%7Bi%7D%3Dg%28c%2Cs_%7Bi-1%7D%2CE_%7Bi-1%7D%29)

表达解码器隐藏层变换：直到解码解出“_EOS”，标志着解码的结束。

### RNN的两个主要问题

1. 后起之秀的新模型的崛起，比如经过特殊改造的CNN模型，以及最近特别流行的Transformer，这些后起之秀尤其是Transformer的应用效果相比RNN来说，目前看具有明显的优势。
2. 另外一个严重阻碍RNN将来继续走红的问题是：RNN本身的序列依赖结构对于大规模并行计算来说相当之不友好。通俗点说，就是RNN很难具备高效的并行计算能力，这个乍一看好像不是太大的问题，其实问题很严重。



![img](https://pic1.zhimg.com/80/v2-41f4da9b0afcfbe39bff0ff3f8097760_hd.jpg)

###  self-attention

1. **微观角度看自注意力机制**

   计算自注意力的第一步就是从每个编码器的输入向量（每个单词的词向量）中生成三个向量。也就是说对于每个单词，我们创造一个查询向量、一个键向量和一个值向量。这三个向量是通过词嵌入与三个权重矩阵后相乘创建的。

![img](https://pic2.zhimg.com/80/v2-bac717483cbeb04d1b5ef393eb87a16d_hd.jpg)

X1与WQ权重矩阵相乘得到q1, 就是与这个单词相关的查询向量。最终使得输入序列的每个单词的创建一个查询向量、一个键向量和一个值向量。

2. **什么是查询向量、键向量和值向量向量**

   计算自注意力的第二步是计算得分。假设我们在为这个例子中的第一个词“Thinking”计算自注意力向量，我们需要拿输入句子中的每个单词对“Thinking”打分。这些分数决定了在编码单词“Thinking”的过程中有多重视句子的其它部分。

   

   这些分数是通过打分单词（所有输入句子的单词）的键向量与“Thinking”的查询向量相点积来计算的。所以如果我们是处理位置最靠前的词的自注意力的话，第一个分数是q1和k1的点积，第二个分数是q1和k2的点积。

   ![img](https://pic2.zhimg.com/80/v2-373fb39e650fa85976bbb6eaf67b31ed_hd.jpg)       第三步和第四步是将分数除以8(8是论文中使用的键向量的维数64的平方根，这会让梯度更稳定。这里也可以使用其它值，8只是默认值)	，然后通过softmax传递结果。softmax的作用是使所有单词的分数归一化，得到的分数都是正值且和为1。

   

   ![preview](https://pic2.zhimg.com/v2-5591c2b55d9e31e744f884bacf959b45_r.jpg)

   这个softmax分数决定了每个单词对编码当下位置（“Thinking”）的贡献。显然，已经在这个位置上的单词将获得最高的softmax分数，但有时关注另一个与当前单词相关的单词也会有帮助。![img](https://pic4.zhimg.com/80/v2-609de8f8f8e628e6a9ca918230c70d67_hd.jpg)

   第五步是将每个值向量乘以softmax分数(这是为了准备之后将它们求和)。这里的直觉是希望关注语义上相关的单词，并弱化不相关的单词(例如，让它们乘以0.001这样的小数)。

   

   第六步是对加权值向量求和（译注：自注意力的另一种解释就是在编码某个单词时，就是将所有单词的表示（值向量）进行加权求和，而权重是通过该词的表示（键向量）与被编码词表示（查询向量）的点积并通过softmax得到。），然后即得到自注意力层在该位置的输出(在我们的例子中是对于第一个单词)。

## Bert

《BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding》

这个题目有五个关键词，分别是 Pre-training、Deep、Bidirectional、Transformers、和 Language Understanding。其中 pre-training 的意思是，作者认为，确实存在通用的语言模型，先用文章预训练通用模型，然后再根据具体应用，用 supervised 训练数据，精加工（fine tuning）模型，使之适用于具体应用。为了区别于针对语言生成的 Language Model，作者给通用的语言模型，取了一个名字，叫语言表征模型 Language Representation Model。



### 预训练任务1：Masked Language Model

现有的语言模型的问题在于，没有同时利用到Bidirectional信息，现有的语言模型例如ELMo号称是双向LM(BiLM)，但是实际上是两个单向RNN构成的语言模型的拼接，如下图所示![img](https://pic3.zhimg.com/80/v2-5686dfb7db40ae74003e42060b7b304a_hd.jpg)

因为语言模型本身的定义是计算句子的概率：![[公式]](https://www.zhihu.com/equation?tex=p%28S%29+%3D+p%28w_1%2Cw_2%2Cw_3%2C...%2Cw_m%29%3Dp%28w_1%29p%28w_2%7Cw_1%29p%28w_3%7Cw_1%2Cw_2%29...p%28w_m%7Cw_1%2Cw_2%2C...%2Cw_%7Bm-1%7D%29%5C%5C+%3D%5Cprod_%7Bi%3D1%7D%5E%7Bm%7Dp%28w_i%7Cw_1%2Cw_2%2C...%2Cw_%7Bi-1%7D%29+%5Ctag%7B2%7D)

前向RNN构成的语言模型计算的是：![[公式]](https://www.zhihu.com/equation?tex=p%28w_1%2Cw_2%2Cw_3%2C...%2Cw_m%29+%3D%5Cprod_%7Bi%3D1%7D%5E%7Bm%7Dp%28w_i%7Cw_1%2Cw_2%2C...%2Cw_%7Bi-1%7D%29+%5Ctag%7B3%7D)

也就是当前词的概率只依赖前面出现词的概率。

而后向RNN构成的语言模型计算的是：![[公式]](https://www.zhihu.com/equation?tex=p%28w_1%2Cw_2%2Cw_3%2C...%2Cw_m%29+%3D%5Cprod_%7Bi%3D1%7D%5E%7Bm%7Dp%28w_i%7Cw_%7Bi%2B1%7D%2Cw_%7Bi%2B2%7D%2C...%2Cw_%7Bm%7D%29+%5Ctag%7B4%7D)

BERT提出了Masked Language Model，也就是随机去掉句子中的部分token，然后模型来预测被去掉的token是什么。这样实际上已经不是传统的神经网络语言模型(类似于生成模型)了，而是单纯作为分类问题，根据这个时刻的hidden state来预测这个时刻的token应该是什么，而不是预测下一个时刻的词的概率分布了。

这里的操作是随机mask语料中15%的token，然后预测masked token，那么masked token 位置输出的final hidden vectors喂给softmax网络即可得到masked token的预测结果。这样操作存在一个问题，fine-tuning的时候没有[MASK] token，因此存在pre-training和fine-tuning之间的mismatch，为了解决这个问题，采用了下面的策略：

- 80%的时间中：将选中的词用[MASK]token来代替，例如

```text
my dog is hairy → my dog is [MASK]
```

- 10%的时间中：将选中的词用任意的词来进行代替，例如

```text
my dog is hairy → my dog is apple
```

- 10%的时间中：选中的词不发生变化，例如

```text
my dog is hairy → my dog is hairy
```

这样存在另一个问题在于在训练过程中只有15%的token被预测，正常的语言模型实际上是预测每个token的，因此Masked LM相比正常LM会收敛地慢一些，后面的实验也的确证实了这一点。

**pytorch版代码如下**

```python
    def random_word(self, sentence):
        tokens = sentence.split()
        output_label = []

        for i, token in enumerate(tokens):
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[i] = self.vocab.mask_index

                # 10% randomly change token to random token
                elif prob < 0.9:
                    tokens[i] = random.randrange(len(self.vocab))

                # 10% randomly change token to current token
                else:
                    tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)

                output_label.append(self.vocab.stoi.get(token, self.vocab.unk_index))

            else:
                tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)
                output_label.append(0)

        return tokens, output_label
```



### Next Sentence Prediction

很多需要解决的NLP tasks依赖于句子间的关系，例如问答任务等，这个关系语言模型是获取不到的，因此将下一句话预测作为了第二个预训练任务。该任务的训练语料是两句话，来预测第二句话是否是第一句话的下一句话，如下所示

![img](https://pic1.zhimg.com/80/v2-00c509eefede31c6ff179f8740ecfd7c_hd.jpg)

而最终该任务得到了97%-98%的准确度。



### fine-tune

这里fine-tuning之前对模型的修改非常简单，例如针对sequence-level classification problem(情感分析)，取第一个token的输出表示，喂给一个softmax层得到分类结果输出；对于token-level classification(例如NER)，取所有token的最后层transformer输出，喂给softmax层做分类。

总之不同类型的任务需要对模型做不同的修改，但是修改都是非常简单的，最多加一层MLP即可。如下图所示

![img](https://pic3.zhimg.com/80/v2-a8de0f4b233bd5d267737689db32388e_hd.jpg)

### 模型结构

![img](https://pic1.zhimg.com/80/v2-d942b566bde7c44704b7d03a1b596c0c_hd.jpg)

对比OpenAI GPT(Generative pre-trained transformer)，BERT是双向的Transformer block连接；就像单向rnn和双向rnn的区别，直觉上来讲效果会好一些。

对比ELMo，虽然都是“双向”，但目标函数其实是不同的。ELMo是分别以![[公式]](https://www.zhihu.com/equation?tex=P%28w_i%7C+w_1%2C+...w_%7Bi-1%7D%29) 和 ![[公式]](https://www.zhihu.com/equation?tex=P%28w_i%7Cw_%7Bi%2B1%7D%2C+...w_n%29) 作为目标函数，独立训练处两个representation然后拼接，而BERT则是以 ![[公式]](https://www.zhihu.com/equation?tex=P%28w_i%7Cw_1%2C++...%2Cw_%7Bi-1%7D%2C+w_%7Bi%2B1%7D%2C...%2Cw_n%29) 作为目标函数训练LM。

### Embedding

[***三种embedding详解***](https://cloud.tencent.com/developer/article/1460597)

#### position embedding

![](D:\md_images\position embedding.jpg)



这里的Embedding由三种Embedding求和而成：![img](https://pic2.zhimg.com/80/v2-11505b394299037e999d12997e9d1789_hd.jpg)

- Token Embeddings是词向量，第一个单词是CLS标志，可以用于之后的分类任务
- Segment Embeddings用来区别两种句子，因为预训练不光做LM还要做以两个句子为输入的分类任务
- Position Embeddings和之前文章中的Transformer不一样，不是三角函数而是学习出来的
- CLS：每个序列的第一个 token 始终是特殊分类嵌入（special classification embedding），即 CLS。对应于该 token 的最终隐藏状态（即，Transformer的输出）被用于分类任务的聚合序列表示。如果没有分类任务的话，这个向量是被忽略的。
- SEP：用于分隔一对句子的特殊符号。有两种方法用于分隔句子：第一种是使用特殊符号 SEP；第二种是添加学习句子 A 嵌入到第一个句子的每个 token 中，句子 B 嵌入到第二个句子的每个 token 中。如果是单个输入的话，就只使用句子 A 。

## XLNet

### 自回归语言模型

在ELMO／BERT出来之前，大家通常讲的语言模型其实是根据上文内容预测下一个可能跟随的单词，就是常说的自左向右的语言模型任务，或者反过来也行，就是根据下文预测前面的单词，__这种类型的LM被称为自回归语言模型__。GPT 就是典型的自回归语言模型。ELMO尽管看上去利用了上文，也利用了下文，但是本质上仍然是自回归LM，这个跟模型具体怎么实现有关系。ELMO是做了两个方向（从左到右以及从右到左两个方向的语言模型），但是是分别有两个方向的自回归LM，然后把LSTM的两个方向的隐节点状态拼接到一起，来体现双向语言模型这个事情的。所以其实是两个自回归语言模型的拼接，本质上仍然是自回归语言模型。

自回归语言模型有优点有缺点，缺点是只能利用上文或者下文的信息，不能同时利用上文和下文的信息，当然，貌似ELMO这种双向都做，然后拼接看上去能够解决这个问题，因为融合模式过于简单，所以效果其实并不是太好。它的优点，其实跟下游NLP任务有关，比如生成类NLP任务，比如文本摘要，机器翻译等，在实际生成内容的时候，就是从左向右的，自回归语言模型天然匹配这个过程。而Bert这种DAE模式，在生成类NLP任务中，就面临训练过程和应用过程不一致的问题，导致生成类的NLP任务到目前为止都做不太好。

### 自编码语言模型

自回归语言模型只能根据上文预测下一个单词，或者反过来，只能根据下文预测前面一个单词。相比而言，Bert通过在输入X中随机Mask掉一部分单词，然后预训练过程的主要任务之一是根据上下文单词来预测这些被Mask掉的单词。那些被Mask掉的单词就是在输入侧加入的所谓噪音。类似Bert这种预训练模式，被称为DAE LM。

XLNet的出发点就是：能否融合自回归LM和DAE LM两者的优点。就是说如果站在自回归LM的角度，如何引入和双向语言模型等价的效果；如果站在DAE LM的角度看，它本身是融入双向语言模型的，如何抛掉表面的那个[Mask]标记，让预训练和Fine-tuning保持一致。



### XLNET主要改进

Bert这种自编码语言模型的好处是：能够同时利用上文和下文，所以信息利用充分。对于很多NLP任务而言，典型的比如阅读理解，在解决问题的时候，是能够同时看到上文和下文的，所以当然应该把下文利用起来。在Bert原始论文中，与GPT1.0的实验对比分析也可以看出来，BERT相对GPT 1.0的性能提升，主要来自于双向语言模型与单向语言模型的差异。

GPT 2.0不信这个邪，坚持沿用GPT 1.0 单向语言模型的旧瓶，装进去了更高质量更大规模预训练数据的新数据，而它的实验结果也说明了，如果想改善预训练语言模型，走这条扩充预序列模型训练数据的路子，是个多快好但是不省钱的方向。

- bert缺点：

Bert的自编码语言模型也有对应的缺点，就是XLNet在文中指出的，第一个预训练阶段因为采取引入[Mask]标记来Mask掉部分单词的训练模式，而Fine-tuning阶段是看不到这种被强行加入的Mask标记的，所以两个阶段存在使用模式不一致的情形，这可能会带来一定的性能损失；另外一个是，Bert在第一个预训练阶段，假设句子中多个单词被Mask掉，这些被Mask掉的单词之间没有任何关系，是条件独立的，而有时候这些单词之间是有关系的

- XLNET做法

___Contenxt_before___  prediction  ___Context_after___

目标：在Contenxt_before中揉入context_after的内容。

比如包含单词Ti的当前输入的句子X，由顺序的几个单词构成，比如x1,x2,x3,x4四个单词顺序构成。我们假设，其中，要预测的单词Ti是x3，位置在Position 3，要想让它能够在上文Context_before中，也就是Position 1或者Position 2的位置看到Position 4的单词x4。可以这么做：假设我们固定住x3所在位置，就是它仍然在Position 3，之后随机排列组合句子中的4个单词，在随机排列组合后的各种可能里，再选择一部分作为模型预训练的输入X。比如随机排列组合后，抽取出x4,x2，x3,x1这一个排列组合作为模型的输入X。于是，x3就能同时看到上文x2，以及下文x4的内容了。这就是XLNet的基本思想，所以说，看了这个就可以理解上面讲的它的初衷了吧：看上去仍然是个自回归的从左到右的语言模型，但是其实通过对句子中单词排列组合，把一部分Ti下文的单词排到Ti的上文位置中，于是，就看到了上文和下文，但是形式上看上去仍然是从左到右在预测后一个单词。

x1->x2->x3->x4      ---->    x2->x4->x3->x1

___“双流自注意力机制”___:

其实就是用来代替Bert的那个[Mask]标记的，因为XLNet希望抛掉[Mask]标记符号，但是比如知道上文单词x1,x2，要预测单词x3，此时在x3对应位置的Transformer最高层去预测这个单词，但是输入侧不能看到要预测的单词x3，Bert其实是直接引入[Mask]标记来覆盖掉单词x3的内容的，等于说[Mask]是个通用的占位符号。而XLNet因为要抛掉[Mask]标记，但是又不能看到x3的输入，于是Query流，就直接忽略掉x3输入了，只保留这个位置信息，用参数w来代表位置的embedding编码。其实XLNet只是扔了表面的[Mask]占位符号，内部还是引入Query流来忽略掉被Mask的这个单词。和Bert比，只是实现方式不同而已。

