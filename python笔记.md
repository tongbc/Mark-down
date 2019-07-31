# python集合
- np.random.permutation
```python
import numpy as np
np.random.seed(5)
a = np.arange(10)
permusion = np.random.permutation(a)
print(permusion)
```
#### numpy数组添加
```python
import numpy as np
embeds = np.asarray([[1,2],[2,3],[3,4]])
temp = np.asarray([5,5])
embeds += temp
print(embeds)
```
#### Math.ceil
```python
import math
print(math.ceil(12.4))
##->13
```
#### Pandas sample
```python
data1 = un_data.loc[(data["CRASHSEV"] == 1)].sample(frac=1).iloc[:10000, :]
##frac是要返回的比例，0.3的话就是返回百分之30的数据
```
#### Pandas数据清洗之缺失数据
```python
import numpy as np
from numpy import nan
import pandas as pd
data=pd.DataFrame(np.arange(3,19,1).reshape(4,4),index=list('abcd'))
print(data)
data.iloc[0:2,0:3]=nan
print(data)
print(data.fillna(0))   ### 用0填充缺失数据
###
      0     1     2   3
a   0.0   0.0   0.0   6
b   0.0   0.0   0.0  10
c  11.0  12.0  13.0  14
d  15.0  16.0  17.0  18
###
```
#### Pandas数据清洗之缺失数据
```python
def one(a,*b):
    """a是一个普通传入参数，*b是一个非关键字星号参数"""
    print(b)
one(1,2,3,4,5,6)
#--------
def two(a=1,**b):
    """a是一个普通关键字参数，**b是一个关键字双星号参数"""
    print(b)
two(a=1,b=2,c=3,d=4,e=5,f=6)
###
第一个输出为：
(2, 3, 4, 5, 6)
第二个输出为：
{'b': 2, 'c': 3, 'e': 5, 'f': 6, 'd': 4}
###
```
#### embedding dict最牛逼形成方法
```python
EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))
```
#### python format用法，
```python
age = "5"
name = "bobby"
print(f'{name} is {age} years old')

f'''He said his name is {name.upper()}and he is {6 * seven} years old.'''
##'He said his name is FRED and he is 42 years old.'

```
#### python string.punctuation用法，
```python
print(string.punctuation)
#!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
```

#### numpy np.newaxis用法
```python
x1 = np.array([1, 2, 3, 4, 5]) 
# the shape of x1 is (5,) 
x1_new = x1[:, np.newaxis] 
# now, the shape of x1_new is (5, 1) 
# array([[1], 
# [2], 
# [3], 
# [4], 
# [5]]) 
x1_new = x1[np.newaxis,:] 
# now, the shape of x1_new is (1, 5) 
# array([[1, 2, 3, 4, 5]])
```

#### multiprocessing.Pool() 用法
```python
def load_data(data):
    return pd.read_csv(data)

with multiprocessing.Pool() as pool:
    train, test, sub = pool.map(load_data, ['../input/train.csv', '../input/test.csv', '../input/sample_submission.csv'])
```

#### collections.defaultdict(list) 用法
```python
targets = collections.defaultdict(list)
```
初始化字典value为list

#### df.target.value_counts() 用法

```python
df = pd.DataFrame([["a1",5],["a2",6]],columns=["a","b"])
df.b.value_counts()

6    1
5    1
```

#### np.argsort(lis) 用法

从小到大返回index

```python
lis = [2,3,1]
np.argsort(lis)[::-1]

array([1, 0, 2])
```

#### sns.barplot

```py
sns.barplot(y=mark, x=x, orient='h')
```

#### itertools.product 用法

从小到大返回index

~~~python
lis = [[1,2,3],[3,6,7]]
for temp in itertools.product(*lis):
    print((temp))
    
```
(1, 3)
(1, 6)
(1, 7)
(2, 3)
(2, 6)
(2, 7)
(3, 3)
(3, 6)
(3, 7)
```
返回排列组合tuple
~~~

#### 

#### contextmanager 用法

计算函数的时间

```python
from contextlib import contextmanager
import time
@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title,time.time()-t0))
   
with timer("test"):
    time.sleep(1)
## test - done in 1s

```

