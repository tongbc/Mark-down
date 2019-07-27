# LeetCode笔记


## 204. 计数质数
### 题目描述

统计所有小于非负整数 n 的质数的数量。
### 解法
```python
class Solution(object):
    def countPrimes(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n <3:
            return 0
        prim = [True]*n
        prim[0],prim[1] = False,False
        for i in range(2,int(n**0.5)+1):
            if prim[i]:
                prim[i*i:n:i] = [False]*len(prim[i*i:n:i])
        return sum(prim)
```

如果自身是质数，则从他的平方开始，到n，把所有为i的倍数都变为非质数。

##  239 滑动窗口最大值

###  题目描述

给定一个数组 *nums*，有一个大小为 *k* 的滑动窗口从数组的最左侧移动到数组的最右侧。你只可以看到在滑动窗口 *k* 内的数字。滑动窗口每次只向右移动一位。

返回滑动窗口最大值。

###  解法

主要思路：利用deque双向队列，保证K的长度的deque，左侧为最多，右侧保证最小，下降的趋势，跟随窗口移动一起移动。

```python
from collections import deque
class Solution:
    def maxSlidingWindow(self, nums, k):
        res = []
        bigger = deque()
        for i, n in enumerate(nums):
            # make sure the rightmost one is the smallest
            while bigger and nums[bigger[-1]] <= n:
                bigger.pop()

            # add in
            bigger += [i]

            # make sure the leftmost one is in-bound
            if i - bigger[0] >= k:
                bigger.popleft()

            # if i + 1 < k, then we are initializing the bigger array
            if i + 1 >= k:
                res.append(nums[bigger[0]])
        return res
```

##  189.旋转数组

### 题目描述

给定一个数组，将数组中的元素向右移动 *k* 个位置，其中 *k* 是非负数。

###  解法

先整体翻转，再0->k-1,k->n翻转

```python
class Solution(object):
    def rotate(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: None Do not return anything, modify nums in-place instead.
        """
        k = k%len(nums)
        self.rev(nums,0,len(nums)-1)
        self.rev(nums,0,k-1)
        self.rev(nums,k,len(nums)-1)
        
    
    def rev(self,nums,a,b):
        while(b>a):
            temp = nums[a]
            nums[a] = nums[b]
            nums[b] = temp
            b-=1
            a+=1
```



##  202.快乐数

###  题目描述

编写一个算法来判断一个数是不是“快乐数”。

一个“快乐数”定义为：对于一个正整数，每一次将该数替换为它每个位置上的数字的平方和，然后重复这个过程直到这个数变为 1，也可能是无限循环但始终变不到 1。如果可以变为 1，那么这个数就是快乐数。

###  解题思路

按照思路写，设定set，存储计算过得数字，如重新出现，则死循环，false，得到1时返回True

```python
class Solution(object):
    def isHappy(self, n):
        """
        :type n: int
        :rtype: bool
        """
        a = set([])
        while(n not in a):
            a.add(n)
            num = 0
            while(n>0):
                remain = n%10
                num += remain**2
                n = n/10
            if num==1:
                return True
            else:
                n = num
        return False
```

##  209.长度最小的子数组

### 题目描述

给定一个含有 **n** 个正整数的数组和一个正整数 **s ，**找出该数组中满足其和 **≥ s** 的长度最小的连续子数组**。**如果不存在符合条件的连续子数组，返回 0。

### 解题思路

用数组的元祖，表示在i的位置上，最接近s但小于s的数值以及长度，超过时找该点在右边时最短长度。

```python
class Solution(object):
    def minSubArrayLen(self, s, nums):
        """
        :type s: int
        :type nums: List[int]
        :rtype: int
        """
        res = []
        mark = ()
        if not nums:
            return 0
        if nums[0]>=s:
            return 1
        else:
            mark=(1,nums[0])
        for i in range(1,len(nums)):
            if nums[i]+mark[1]>=s:
                temp = nums[i]+mark[1]
                count = 0
                while(temp>=s):
                    temp-=nums[i-mark[0]+count]
                    count+=1
                res.append(mark[0]+2-count)
                mark=((1+mark[0]-count,temp))
            else:
                mark=((1+mark[0],nums[i]+mark[1]))
        if not res:
            return 0
        return min(res)
```

##  169.求众数

### 题目描述

给定一个大小为 *n* 的数组，找到其中的众数。众数是指在数组中出现次数**大于** `⌊ n/2 ⌋` 的元素。

你可以假设数组是非空的，并且给定的数组总是存在众数。

### 解题思路

因为大与一半的数量存在，所以当重合时候加1，不相等时候减一，减到0把标记归None，最后留下的就是answer。

```python
class Solution(object):
    def majorityElement(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        major = None
        c = 0
        for num in nums:
            if not major:
                major = num
                c += 1
            if major == num:
                c += 1
            elif major != num:
                c -= 1
                if c == 0:
                    major = None
        return major
```

##  171.Excel表列序号

###  题目描述

给定一个Excel表格中的列名称，返回其相应的列序号。

例如

```
A -> 1
    B -> 2
    C -> 3
    ...
    Z -> 26
    AA -> 27
    AB -> 28 
    ...
```

### 解题思路

每一位，都是26^index*(与A的asc码加一)

```python
class Solution(object):
    def titleToNumber(self,s):
        s = s[::-1]
        sum = 0
        for exp, char in enumerate(s):
            sum += (ord(char) - 65 + 1) * (26 ** exp)
        return sum
```

##  743.网络延迟时间

### 题目描述

有 `N` 个网络节点，标记为 `1` 到 `N`。

给定一个列表 `times`，表示信号经过**有向**边的传递时间。 `times[i] = (u, v, w)`，其中 `u` 是源节点，`v` 是目标节点， `w` 是一个信号从源节点传递到目标节点的时间。

现在，我们向当前的节点 `K` 发送了一个信号。需要多久才能使所有节点都收到信号？如果不能使所有节点收到信号，返回 `-1`。

**注意:**

1. `N` 的范围在 `[1, 100]` 之间。
2. `K` 的范围在 `[1, N]` 之间。
3. `times` 的长度在 `[1, 6000]` 之间。
4. 所有的边 `times[i] = (u, v, w)` 都有 `1 <= u, v <= N` 且 `0 <= w <= 100`。

###  解题思路

BFS：将线路和时间都存在以出发点为key的字典里，然后分层遍历，时间短的直接代替

```python
import collections
class Solution:
    def networkDelayTime(self, times, N, K):
        t, graph, q = [0] + [float("inf")] * N, collections.defaultdict(list), collections.deque([(0, K)])
        for u, v, w in times:
            graph[u].append((v, w))
        while q:
            time, node = q.popleft()
            if time < t[node]:
                t[node] = time
                for v, w in graph[node]:
                    q.append((time + w, v))
        mx = max(t)
        return mx if mx < float("inf") else -1

```

##  166.分数到小数

###  题目描述

给定两个整数，分别表示分数的分子 numerator 和分母 denominator，以字符串形式返回小数。

如果小数部分为循环小数，则将循环的部分括在括号内。

**示例 1:**

```
输入: numerator = 1, denominator = 2
输出: "0.5"
```

### 解题思路

每次都取整与余数，余数每一位放在stack里，余数乘以10再继续，知道出现过一次，就开始循环。

```python
class Solution(object):
    def fractionToDecimal(self, numerator, denominator):
        """
        :type numerator: int
        :type denominator: int
        :rtype: str
        """
        n,b = divmod(abs(numerator),abs(denominator))
        sign = "-" if numerator*denominator<0 else ""
        result =[sign+str(n),"."]
        stack = {}
        count = 0
        while(b not in stack.keys()):
            stack[b] = count
            count +=1 
            n,b = divmod(b*10,abs(denominator))
            result.append(str(n))
        idx = stack[b]
        result.insert(idx+2,"(")
        result.append(')')
        return ''.join(result).replace('(0)', '').rstrip('.')
```

##  14.最长公共前缀

### 题目描述

编写一个函数来查找字符串数组中的最长公共前缀。

如果不存在公共前缀，返回空字符串 `""`。

### 	解题思路

python根据asc码排序

```python
class Solution(object):
    def longestCommonPrefix(self, strs):
        """
        :type strs: List[str]
        :rtype: str
        """
        if not strs:return ""
        s1 = min(strs)
        s2 = max(strs)
        for i,x in enumerate(s1):
            if x!=s2[i]:
                return s1[:i]
        return s1
```



## 91.解码方法

### 题目描述

一条包含字母 A-Z 的消息通过以下方式进行了编码：

'A' -> 1
'B' -> 2
...
'Z' -> 26
给定一个只包含数字的非空字符串，请计算解码方法的总数。

### 解题思路

dp动态规划，dp[i] = dp[i-1] + dp[i-2]

```python
class Solution(object):
    def numDecodings(self, s):
        """
        :type s: str
        :rtype: int
        """
        if s=="":return 0
        dp = [0 for i in range(len(s)+1)]
        dp[0] = 1
        for i in range(1,len(s)+1):
            if s[i-1]!="0":
                dp[i]+=dp[i-1]
            if i!=1 and s[i-2:i]<"27" and s[i-2:i]>"09":
                dp[i]+=dp[i-2]
        return dp[len(s)]
```

"126"1,2,3



## 34. 在排序数组中查找元素的第一个和最后一个位置

### 题目描述

给定一个按照升序排列的整数数组 nums，和一个目标值 target。找出给定目标值在数组中的开始位置和结束位置。

你的算法时间复杂度必须是 O(log n) 级别。

如果数组中不存在目标值，返回 [-1, -1]。

示例 1:

输入: nums = [5,7,7,8,8,10], target = 8
输出: [3,4]

### 解题思路

- 二分

```python
class Solution(object):
    def searchRange(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        st, end = -1, -1
        l, r = 0, len(nums) - 1
        while l <= r:
            mid = (l + r) // 2
            if nums[mid] < target:
                l += 1
            elif nums[mid] > target:
                r -= 1
            else:
                if mid - 1 < 0 or nums[mid - 1] < nums[mid]:
                    st = mid
                    break
                else:
                    r -= 1
        r = len(nums)-1
        while l <= r:
            mid = (l + r) // 2
            if nums[mid] < target:
                l += 1
            elif nums[mid] > target:
                r -= 1
            else:
                if mid + 1 >len(nums)-1 or nums[mid + 1] > nums[mid]:
                    end = mid
                    break
                else:
                    l += 1
        return[st,end]
```

## 402.移掉k位数字

### 题目描述

给定一个以字符串表示的非负整数 num，移除这个数中的 k 位数字，使得剩下的数字最小。

注意:

num 的长度小于 10002 且 ≥ k。
num 不会包含任何前导零。
示例 1 :

输入: num = "1432219", k = 3
输出: "1219"
解释: 移除掉三个数字 4, 3, 和 2 形成一个新的最小的数字 1219。

### 解题思路

每次移除掉第一个num[i]>num[i+1]的元素

stack版本

```python
class Solution(object):
    def removeKdigits(self, num, k):
        """
        :type num: str
        :type k: int
        :rtype: str
        """
        stack = []
        for idx in range(len(num)):
            while stack and k > 0:
                if num[idx] < stack[-1]:
                    k -= 1
                    stack.pop()
                else:
                    break
            stack.append(num[idx])
        while (k != 0):
            stack.pop()
            k -= 1
        while stack:
            if stack[0] == '0':
                stack = stack[1:]
            else:
                break

        return ''.join(stack) or '0'
```

## 22. 括号生成

### 题目描述

给出 n 代表生成括号的对数，请你写出一个函数，使其能够生成所有可能的并且有效的括号组合。

例如，给出 n = 3，生成结果为：

[
  "((()))",
  "(()())",
  "(())()",
  "()(())",
  "()()()"
]

### 解题思路

dfs,左括号，右括号数量计数

```python
class Solution(object):
    def generateParenthesis(self, n):
        """
        :type n: int
        :rtype: List[str]
        """
        res = []
        self.part("(",1,0,n,res)
        return res
        
    def part(self,pre,a,b,n,res):
        if n==a and a==b:
            res.append(pre)
        else:
            if n>a:
                self.part(pre+"(",a+1,b,n,res)
            if a>b:
                self.part(pre+")",a,b+1,n,res)
            
    
```

## 40. 组合总和 II

### 题目描述

给定一个数组 candidates 和一个目标数 target ，找出 candidates 中所有可以使数字和为 target 的组合。

candidates 中的每个数字在每个组合中只能使用一次。

说明：

所有数字（包括目标数）都是正整数。
解集不能包含重复的组合。 
示例 1:

输入: candidates = [10,1,2,7,6,1,5], target = 8,
所求解集为:
[
  [1, 7],
  [1, 2, 5],
  [2, 6],
  [1, 1, 6]
]

### 解题思路

dfs,先排序，每一大轮次，不加重复的数字，如果数字大于剩下的target，则break，因为后面的也肯定越界。

```python
class Solution(object):
    def combinationSum2(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """

        res = []
        cand = sorted(candidates)
        self.part(0, [], cand, target, res)
        return res

    def part(self, i, p, nums, target, res):
        if target == 0:
            res.append(p)
        for j in range(i, len(nums)):
            if j > i and nums[j] == nums[j - 1]:
                continue
            if nums[j] > target:
                break
            self.part(j + 1, p + [nums[j]], nums, target - nums[j], res)


```

## 31. 下一个排列

### 题目描述

实现获取下一个排列的函数，算法需要将给定数字序列重新排列成字典序中下一个更大的排列。

如果不存在下一个更大的排列，则将数字重新排列成最小的排列（即升序排列）。

必须原地修改，只允许使用额外常数空间。

以下是一些例子，输入位于左侧列，其相应输出位于右侧列。
1,2,3 → 1,3,2
3,2,1 → 1,2,3
1,1,5 → 1,5,1



### 解题思路

dfs,先排序，每一大轮次，不加重复的数字，如果数字大于剩下的target，则break，因为后面的也肯定越界。

```python
class Solution(object):
    def nextPermutation(self, nums):
        """
        :type nums: List[int]
        :rtype: None Do not return anything, modify nums in-place instead.
        """
        l = len(nums)
        for i in range(l - 2, -1, -1):
            if nums[i] >= nums[i + 1]:
                continue
            else:
                for j in range(l-1, i,-1 ):
                    if nums[j] > nums[i]:
                        self.swap(nums, j, i)
                        self.reverse(nums,i+1,l-1)
                        ## 改进：不需要排序，因为插入的位置
                        # nums[i + 1:] = sorted(nums[i + 1:])
                        return nums
        self.reverse(nums,0,l-1)
        return nums

    def swap(self, nums, a, b):
        temp = nums[a]
        nums[a] = nums[b]
        nums[b] = temp

    def reverse(self,nums,l,r):
        while l < r:
            nums[l],nums[r] = nums[r],nums[l]
            l += 1
            r -= 1
```

​	

## 93. 复原IP

给定一个只包含数字的字符串，复原它并返回所有可能的 IP 地址格式。

示例:

输入: "25525511135"
输出: ["255.255.11.135", "255.255.111.35"]



### 解题思路

dfs

```python
class Solution(object):
    def restoreIpAddresses(self, s):
        """
        :type s: str
        :rtype: List[str]
        """
        res = []
        self.dfs([],s,4,res)
        return res
    def dfs(self,lis,s,target,res):
        if len(s)>target*3 or target>len(s):
            return
        if target==1:
            if int(s)<=255:
                lis.append(s)
                res.append(".".join(lis))
                return
            else:
                return
        if s[0]=="0":
            self.dfs(lis+["0"],s[1:],target-1,res)
            return
        self.dfs(lis+[s[:2]],s[2:],target-1,res)
        if int(s[:3])<=255:
            self.dfs(lis+[s[:3]],s[3:],target-1,res)
```

## 318.最大单词长度乘积

### 题目描述

给定一个字符串数组 words，找到 length(word[i]) * length(word[j]) 的最大值，并且这两个单词不含有公共字母。你可以认为每个单词只包含小写字母。如果不存在这样的两个单词，返回 0。

示例 1:

输入: ["abcw","baz","foo","bar","xtfn","abcdef"]
输出: 16 
解释: 这两个单词为 "abcw", "xtfn"

### 解题思路

最最最重要的收获，二进制，表示set的字符串，每一位ord(c)-ord("a")，变一表示有这个字母。两个字符串不同：

他们的二进制求与，如果不重合的话，为0，求反则为True。

### tag

二进制

```python
class Solution(object):
    def maxProduct(self, words):
        """
        :type words: List[str]
        :rtype: int
        """
        d = {}
        for w in words:
            mask = 0
            for c in set(w):
                mask |= 1<<(ord(c)-ord("a"))
            d[mask] = max(d.get(mask,0),len(w))
        return max([d[x]*d[y] for x in d for y in d if not x&y] or [0])
```

## 322.领钱兑换

### 题目描述

给定不同面额的硬币 coins 和一个总金额 amount。编写一个函数来计算可以凑成总金额所需的最少的硬币个数。如果没有任何一种硬币组合能组成总金额，返回 -1。

示例 1:

输入: coins = [1, 2, 5], amount = 11
输出: 3 
解释: 11 = 5 + 5 + 1
示例 2:

输入: coins = [2], amount = 3
输出: -1

### 解题思路

动态规划，每个位置的最小等于减去coins中其他硬币价值的记录值+1.

他们的二进制求与，如果不重合的话，为0，求反则为True。

### tag

动态规划

```python
class Solution(object):
    def maxProduct(self, words):
        """
        :type words: List[str]
        :rtype: int
        """
        d = {}
        for w in words:
            mask = 0
            for c in set(w):
                mask |= 1<<(ord(c)-ord("a"))
            d[mask] = max(d.get(mask,0),len(w))
        return max([d[x]*d[y] for x in d for y in d if not x&y] or [0])
```

## 324.摆动排序 II

### 题目描述

给定一个无序的数组 nums，将它重新排列成 nums[0] < nums[1] > nums[2] < nums[3]... 的顺序。

示例 1:

输入: nums = [1, 5, 1, 1, 6, 4]
输出: 一个可能的答案是 [1, 4, 1, 5, 1, 6]

### 解题思路

https://leetcode.com/problems/wiggle-sort-ii/discuss/77681/O(n)-time-O(1)-space-solution-with-detail-explanations

### tag

N-th largest num

```python
class Solution {
public:
	void wiggleSort(vector<int>& nums) {
		if (nums.empty()) {
			return;
		}    
		int n = nums.size();
		
		// Step 1: Find the median    		
		vector<int>::iterator nth = next(nums.begin(), n / 2);
		nth_element(nums.begin(), nth, nums.end());
		int median = *nth;

		// Step 2: Tripartie partition within O(n)-time & O(1)-space.    		
		auto m = [n](int idx) { return (2 * idx + 1) % (n | 1); };    		
		int first = 0, mid = 0, last = n - 1;
		while (mid <= last) {
			if (nums[m(mid)] > median) {
				swap(nums[m(first)], nums[m(mid)]);
				++first;
				++mid;
			}
			else if (nums[m(mid)] < median) {
				swap(nums[m(mid)], nums[m(last)]);
				--last;
			}				
			else {
				++mid;
			}
		}
	}    
};
```

## 326.3的幂

### 题目描述

给定一个整数，写一个函数来判断它是否是 3 的幂次方。

**示例 1:**

```
输入: 27
输出: true
```

**示例 2:**

```
输入: 0
输出: false
```

### 解题思路

最大的int三次幂的数字能否整除这个数字

0x7fffffff   是最大的int

### tag

N-th largest num

```python
import math
class Solution(object):
    def isPowerOfThree(self, n):
        """
        :type n: int
        :rtype: bool
        """
        m = (int)(math.pow(3,(int)(math.log(0x7fffffff)/math.log(3))))
        return ( n>0 and  m%n==0)
```

## 328.奇偶链表

### 题目描述

给定一个单链表，把所有的奇数节点和偶数节点分别排在一起。请注意，这里的奇数节点和偶数节点指的是节点编号的奇偶性，而不是节点的值的奇偶性。

请尝试使用原地算法完成。你的算法的空间复杂度应为 O(1)，时间复杂度应为 O(nodes)，nodes 为节点总数。

示例 1:

输入: 1->2->3->4->5->NULL
输出: 1->3->5->2->4->NULL
示例 2:

输入: 2->1->3->5->6->4->7->NULL 
输出: 2->3->6->7->1->5->4->NULL

### 解题思路

两部分

### tag

链表

```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def oddEvenList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if not head or not head.next:
            return head
        a,b,bhead = head,head.next,head.next
        while(b.next and b.next.next):
            temp1,temp2 = b.next,b.next.next
            a.next = temp1
            temp1.next = bhead
            b.next = temp2
            a = temp1
            b = temp2
        if b.next:
            a.next = b.next
            a.next.next = bhead
        return head
            
```

更优解

```python
def oddEvenList(self, head):
    dummy1 = odd = ListNode(0)
    dummy2 = even = ListNode(0)
    while head:
        odd.next = head
        even.next = head.next
        odd = odd.next
        even = even.next
        head = head.next.next if even else None
    odd.next = dummy2.next
    return dummy1.next
```

## 200.岛屿数量

### 题目描述

给定一个由 '1'（陆地）和 '0'（水）组成的的二维网格，计算岛屿的数量。一个岛被水包围，并且它是通过水平方向或垂直方向上相邻的陆地连接而成的。你可以假设网格的四个边均被水包围。

示例 1:

输入:
11110
11010
11000
00000

输出: 1

### 解题思路

dfs,每次把每一部分的岛屿遍历并且标记成"#"，以防以后重复遍历

### tag

DFS

```python
class Solution(object):
    def numIslands(self, grid):
        """
        :type grid: List[List[str]]
        :rtype: int
        """
        count = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] =="1":
                    self.dfs(i,j,grid)
                    count+=1
        return count
    
    def dfs(self,i,j,grid):
        if i<0 or j<0 or i>=len(grid) or j>=len(grid[0]) or grid[i][j] != '1':
            return  
        grid[i][j] = "#"
        self.dfs(i-1,j,grid)
        self.dfs(i+1,j,grid)
        self.dfs(i,j-1,grid)
        self.dfs(i,j+1,grid)

        return 
```

## 334.递增的三元子序列

### 题目描述

给定一个未排序的数组，判断这个数组中是否存在长度为 3 的递增子序列。

数学表达式如下:

如果存在这样的 i, j, k,  且满足 0 ≤ i < j < k ≤ n-1，
使得 arr[i] < arr[j] < arr[k] ，返回 true ; 否则返回 false 。
说明: 要求算法的时间复杂度为 O(n)，空间复杂度为 O(1) 。

示例 1:

输入: [1,2,3,4,5]
输出: true
示例 2:

输入: [5,4,3,2,1]
输出: false

### 解题思路

用m表示之前最小的数，res表示当前的最小上升序列。

### tag

遍历

```python
class Solution(object):
    def increasingTriplet(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        m = 0x7fffffff
        res = []
        for i in range(len(nums)):
            m = min(nums[i],m)
            if len(res)==0:
                res.append(m)
                continue
            if len(res)==1:
                if nums[i]>res[0]:
                    res.append(nums[i])
                else:
                    m = min(m,nums[i])
                    res[0] = m
            elif len(res)==2:
                if nums[i]>res[1]:
                    return True
                else:
                    if nums[i]>m:
                        res[0] = m
                        res[1] = nums[i]
                    else:
                        m = nums[i]
        return False
```

## 334.重新安排行程

### 题目描述

给定一个机票的字符串二维数组 [from, to]，子数组中的两个成员分别表示飞机出发和降落的机场地点，对该行程进行重新规划排序。所有这些机票都属于一个从JFK（肯尼迪国际机场）出发的先生，所以该行程必须从 JFK 出发。

说明:

如果存在多种有效的行程，你可以按字符自然排序返回最小的行程组合。例如，行程 ["JFK", "LGA"] 与 ["JFK", "LGB"] 相比就更小，排序更靠前
所有的机场都用三个大写字母表示（机场代码）。
假定所有机票至少存在一种合理的行程。
示例 1:

输入: [["MUC", "LHR"], ["JFK", "MUC"], ["SFO", "SJC"], ["LHR", "SFO"]]
输出: ["JFK", "MUC", "LHR", "SFO", "SJC"]

### 解题思路

先greedy，如果greedy没有路径，就走另一条路

### tag

greedy+dfs

```python
class Solution(object):
    def findItinerary(self, tickets):
        """
        :type tickets: List[List[str]]
        :rtype: List[str]
        """
        dic = {}
        for i in range(tickets):
            if dic.haskey(tickets[i][0]):
                dic[tickets[i][0]].extend(tickets[i][1])
            else:
                dic[tickets[i][0]] = [tickets[i][1]]
        for i in dic.keys():
            if len(dic[i])>1:
                dic[i].sort()
        res = ["JFK"]
        if len(dic[res[-1]])>0:
            res.append(dic[res[-1]][0])
            dic[res[-2]].remove(dic[res[-2]][0])
        return res
        
```

## 343.整数拆分

### 题目描述

给定一个正整数 n，将其拆分为至少两个正整数的和，并使这些整数的乘积最大化。 返回你可以获得的最大乘积。

示例 1:

输入: 2
输出: 1
解释: 2 = 1 + 1, 1 × 1 = 1。
示例 2:

输入: 10
输出: 36
解释: 10 = 3 + 3 + 4, 3 × 3 × 4 = 36。

### 解题思路

找规律，2 * 2 * 2<3 * 3,so no more than 2s 2    4或4以上的，都可以替代为2,3,1,1 is a waste

### tag



```python
class Solution(object):
    def integerBreak(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n==2:
            return 1
        if n==3:
            return 2
        num = 1
        while(n>4):
            num*=3
            n-=3
        num*=n
        return num
```

## 338.比特位计数

### 题目描述

给定一个非负整数 num。对于 0 ≤ i ≤ num 范围中的每个数字 i ，计算其二进制数中的 1 的数目并将它们作为数组返回。

示例 1:

输入: 2
输出: [0,1,1]
示例 2:

输入: 5
输出: [0,1,1,2,1,2]

### 解题思路

动态规划问题：当i为偶数时，为dp[i/2]，如8为4左移一位，但是1的个数不变，当i为奇数时，为dp[i-1]+1，因为第一位要加个1.

### tag

dp

```python
class Solution(object):
    def integerBreak(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n==2:
            return 1
        if n==3:
            return 2
        num = 1
        while(n>4):
            num*=3
            n-=3
        num*=n
        return num
```

## 413.等差数列划分

### 题目描述

数组 A 包含 N 个数，且索引从0开始。数组 A 的一个子数组划分为数组 (P, Q)，P 与 Q 是整数且满足 0<=P<Q<N 。

如果满足以下条件，则称子数组(P, Q)为等差数组：

元素 A[P], A[p + 1], ..., A[Q - 1], A[Q] 是等差的。并且 P + 1 < Q 。

函数要返回数组 A 中所有为等差数组的子数组个数。

 

示例:

A = [1, 2, 3, 4]

返回: 3, A 中有三个子等差数组: [1, 2, 3], [2, 3, 4] 以及自身 [1, 2, 3, 4]。

### 解题思路

动态规划问题：比如原本是 [1,2,3]，现在又来了4，首先要加一，因为多了长度为4的等差数列，其次多了[2,3,4]的组合，cur是包括末尾的增加的次数，

### tag

dp

```python
class Solution(object):
    def numberOfArithmeticSlices(self, A):
        curr, sum = 0, 0
        for i in range(2,len(A)):
            if A[i]-A[i-1] == A[i-1]-A[i-2]:
                curr += 1
                sum += curr
            else:
                curr = 0
        return sum
```

## 467.环绕字符串中唯一的子字符串

### 题目描述

把字符串 s 看作是“abcdefghijklmnopqrstuvwxyz”的无限环绕字符串，所以 s 看起来是这样的："...zabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcd....". 

现在我们有了另一个字符串 p 。你需要的是找出 s 中有多少个唯一的 p 的非空子串，尤其是当你的输入是字符串 p ，你需要输出字符串 s 中 p 的不同的非空子串的数目。 

注意: p 仅由小写的英文字母组成，p 的大小可能超过 10000。

 

示例 1:

输入: "a"
输出: 1
解释: 字符串 S 中只有一个"a"子字符。

示例 2:

输入: "cac"
输出: 2
解释: 字符串 S 中的字符串“cac”只有两个子串“a”、“c”。.


示例 3:

输入: "zab"
输出: 6
解释: 在字符串 S 中有六个子串“z”、“a”、“b”、“za”、“ab”、“zab”。.

### 解题思路

动态规划问题：用字典记录，以每个字母结尾的最长连续数组的排列，连续的话就加一，不连续回归1，所有的相加最后得出结果。

### tag

dp

```python
class Solution:
    def findSubstringInWraproundString(self, p: str) -> int:
        res = {i:1 for i in p}
        l  =1
        for i,j in zip(p,p[1:]):
            l = l+1 if (ord(j)-ord(i))%26==1 else 1
            res[j] = max(res[j],l)
        return sum(res.values())
```

## 797.所有可能的路径

### 题目描述

给一个有 n 个结点的有向无环图，找到所有从 0 到 n-1 的路径并输出（不要求按顺序）

二维数组的第 i 个数组中的单元都表示有向图中 i 号结点所能到达的下一些结点（译者注：有向图是有方向的，即规定了a→b你就不能从b→a）空就是没有下一个结点了。

示例:
输入: [[1,2], [3], [3], []] 
输出: [[0,1,3],[0,2,3]] 
解释: 图是这样的:
0--->1
|    |
v    v
2--->3
这有两条路: 0 -> 1 -> 3 和 0 -> 2 -> 3.

### 解题思路

dfs,无环，确定开头与结尾，so easy

### tag

dfs

```python
class Solution(object):
    def numberOfArithmeticSlices(self, A):
        curr, sum = 0, 0
        for i in range(2,len(A)):
            if A[i]-A[i-1] == A[i-1]-A[i-2]:
                curr += 1
                sum += curr
            else:
                curr = 0
        return sum
```

## 752.打开转盘锁

### 题目描述

你有一个带有四个圆形拨轮的转盘锁。每个拨轮都有10个数字： '0', '1', '2', '3', '4', '5', '6', '7', '8', '9' 。每个拨轮可以自由旋转：例如把 '9' 变为  '0'，'0' 变为 '9' 。每次旋转都只能旋转一个拨轮的一位数字。

锁的初始数字为 '0000' ，一个代表四个拨轮的数字的字符串。

列表 deadends 包含了一组死亡数字，一旦拨轮的数字和列表里的任何一个元素相同，这个锁将会被永久锁定，无法再被旋转。

字符串 target 代表可以解锁的数字，你需要给出最小的旋转次数，如果无论如何不能解锁，返回 -1。

 

示例 1:

输入：deadends = ["0201","0101","0102","1212","2002"], target = "0202"
输出：6
解释：
可能的移动序列为 "0000" -> "1000" -> "1100" -> "1200" -> "1201" -> "1202" -> "0202"。
注意 "0000" -> "0001" -> "0002" -> "0102" -> "0202" 这样的序列是不能解锁的，
因为当拨动到 "0102" 时这个锁就会被锁定。

### 解题思路

BFS，把所有能走没走过的路遍历，bfs，走过的路加入deadend，

### tag

BFS

```python
class Solution(object):
    def openLock(self, deadends, target):
        """
        :type deadends: List[str]
        :type target: str
        :rtype: int
        """
        moved, q, cnt, move = set(deadends), ["0000"], 0, {str(i): [str((i + 1) % 10), str((i - 1) % 10)] for i in range(10)}

        if "0000" in moved:
            return -1
        while q:
            new = []
            cnt+=1
            for s in q:
                for i,c in enumerate(s):
                    for cur in (s[:i]+move[c][0]+s[i+1:],s[:i]+move[c][1]+s[i+1:]):
                        if cur not in moved:
                            if cur == target:
                                return cnt
                            else:
                                new.append(cur)
                                moved.add(cur)
            q = new
        return -1
```

## 795.区间子数组个数

### 题目描述

给定一个元素都是正整数的数组A ，正整数 L 以及 R (L <= R)。

求连续、非空且其中最大元素满足大于等于L 小于等于R的子数组个数。

例如 :
输入: 
A = [2, 1, 4, 3]
L = 2
R = 3
输出: 3
解释: 满足条件的子数组: [2], [2, 1], [3].
注意:

L, R  和 A[i] 都是整数，范围在 [0, 10^9]。
数组 A 的长度范围在[1, 50000].

### 解题思路

DP问题，读题需要清晰，1.当在左侧时候，可以融入包含左边字母的序列，所以直接加dp，2.当在右边时候，序列需要中断，dp归零 3.当附和条件时候，包含这个数字的，到pre，也就是最后一个不符合的，是i-pre个。最后求和

### tag

DP

```python
class Solution(object):
    def numSubarrayBoundedMax(self, A, L, R):
        """
        :type A: List[int]
        :type L: int
        :type R: int
        :rtype: int
        """
        pre, dp = -1, 0
        res = 0
        for i in range(len(A)):
            if A[i] < L:
                res += dp
            elif A[i] > R:
                dp = 0
                pre = i
            else:
                dp = i - pre
                res += dp
        return res

```

## 756.金字塔转换矩阵

### 题目描述

现在，我们用一些方块来堆砌一个金字塔。 每个方块用仅包含一个字母的字符串表示，例如 “Z”。

使用三元组表示金字塔的堆砌规则如下：

(A, B, C) 表示，“C”为顶层方块，方块“A”、“B”分别作为方块“C”下一层的的左、右子块。当且仅当(A, B, C)是被允许的三元组，我们才可以将其堆砌上。

初始时，给定金字塔的基层 bottom，用一个字符串表示。一个允许的三元组列表 allowed，每个三元组用一个长度为 3 的字符串表示。

如果可以由基层一直堆到塔尖返回true，否则返回false。

示例 1:

输入: bottom = "XYZ", allowed = ["XYD", "YZE", "DEA", "FFF"]
输出: true
解析:
可以堆砌成这样的金字塔:
    A
   / \
  D   E
 / \ / \
X   Y   Z

因为符合('X', 'Y', 'D'), ('Y', 'Z', 'E') 和 ('D', 'E', 'A') 三种规则。

### 解题思路

DFS，把每一层可能的结果排列组合出来，送到下一层dfs里，如果返回True直接返回True，否则继续遍历。

### tag

DFS

```python
class Solution(object):
    def numSubarrayBoundedMax(self, A, L, R):
        """
        :type A: List[int]
        :type L: int
        :type R: int
        :rtype: int
        """
        pre, dp = -1, 0
        res = 0
        for i in range(len(A)):
            if A[i] < L:
                res += dp
            elif A[i] > R:
                dp = 0
                pre = i
            else:
                dp = i - pre
                res += dp
        return res

```

## 767.重构字符串

### 题目描述

给定一个字符串S，检查是否能重新排布其中的字母，使得两相邻的字符不同。

若可行，输出任意可行的结果。若不可行，返回空字符串。

示例 1:

输入: S = "aab"
输出: "aba"
示例 2:

输入: S = "aaab"
输出: ""

### 解题思路

根据最大堆的性质，每次弹出一种字母，随后将数字加一，与字母一同记录，随后下一轮如果还有的话，重新加回heap。保证每两个连续的不出自同一个字母

### tag

Greedy，heap

```python
from collections import Counter
import heapq
class Solution(object):
    def reorganizeString(self, S):
        """
        :type S: str
        :rtype: str
        """
        res = ""
        pq = []
        c = Counter(S)
        for key,value in c.items():
            heapq.heappush(pq,(-value,key))
        p_a,p_b = 0,""
        while(pq):
            a,b = heapq.heappop(pq)
            res += b
            if p_a<0:
                heapq.heappush(pq,(p_a,p_b))
            a += 1
            p_a,p_b = a,b
        if len(res) == len(S):
            return res
        return ""
```

## 745.到达终点数字

### 题目描述

在一根无限长的数轴上，你站在0的位置。终点在target的位置。

每次你可以选择向左或向右移动。第 n 次移动（从 1 开始），可以走 n 步。

返回到达终点需要的最小移动次数。

示例 1:

输入: target = 3
输出: 2
解释:
第一次移动，从 0 到 1 。
第二次移动，从 1 到 3 。
示例 2:

输入: target = 2
输出: 3
解释:
第一次移动，从 0 到 1 。
第二次移动，从 1 到 -1 。
第三次移动，从 -1 到 2 .

### 解题思路

https://www.cnblogs.com/grandyang/p/8456022.html

### tag

math

```python
from collections import deque
class Solution(object):
    def reachNumber(self, target):
        """
        :type target: int
        :rtype: int
        """
        lis = deque([0])
        # lis = [0]
        if target == 0:
            return 0
        l = 1
        step = 1
        while (True):
            for i in range(l):
                temp = lis.popleft()
                if temp + step == target or temp - step == target:
                    return step
                else:
                    lis.append(temp + step)
                    lis.append(temp - step)
            l = len(lis)
            step += 1

```

