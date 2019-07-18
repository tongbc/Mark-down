#  剑指Offer

##  二叉树的深度

###  题目描述：

输入一棵二叉树，求该树的深度。从根结点到叶结点依次经过的结点（含根、叶结点）形成树的一条路径，最长路径的长度为树的深度。

###  解题思路

```python
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def TreeDepth(self, pRoot):
        # write code here
        if not pRoot:
            return 0
        return 1+max(self.dep(pRoot.left),self.dep(pRoot.right))
                   
    def dep(self,root):
        if not root:
            return 0
        return 1+max(self.dep(root.left),self.dep(root.right))            
```

##  扑克顺子

###  题目描述：

几张连续的话贼为顺子，大小王为0，万能代替

### 解题思路

```python
class Solution:
    def IsContinuous(self, numbers):
        # write code here
        count = 0
        lis = set([])
        if not numbers:
            return False
        for num in numbers:
            if num==0:
                count+=1
                continue
            if num in lis:
                return False
            lis.add(num)
        mi = min(lis)
        ma = max(lis)
        return ma-mi-1==len(lis)-2+count or mi==ma
```

