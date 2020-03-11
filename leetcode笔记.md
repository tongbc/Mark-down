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
                for j in range(i * i, n, i):
                	primes[j] = False
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
    
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        m = float("inf")
        dp = [0] + [m]*amount
        for i in range(1,amount+1):
            dp[i] = min(dp[i-c] if i-c>=0 else m for c in coins)+1
        if dp[amount] == m:
            return -1
        else:
            return dp[amount]
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

（1）动态规划问题：当i为偶数时，为dp[i/2]，如8为4左移一位，但是1的个数不变，当i为奇数时，为dp[i-1]+1，因为第一位要加个1.

（2）i&i-1,将i的最右边一个1置为0，随后之前记录下来+1即可。

### tag

dp

```python
class Solution:
    def countBits(self, num: int) -> List[int]:
        res = [0]*(num+1)
        if num == 0:
            return [0]
        for i in range(1,num+1):
            if i%2==0:
                res[i] = res[int(i/2)]
            else:
                res[i] = res[i-1]+1
        return res
    
class Solution:
    def countBits(self, num: int) -> List[int]:
        res = [0]*(num+1)
        if num == 0:
            return [0]
        for i in range(1,num+1):
            res[i] = res[i&(i-1)]+1
        return res
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

## 769.最多能完成排序的块

### 题目描述

数组arr是[0, 1, ..., arr.length - 1]的一种排列，我们将这个数组分割成几个“块”，并将这些块分别进行排序。之后再连接起来，使得连接的结果和按升序排序后的原数组相同。

我们最多能将数组分成多少块？

示例 1:

输入: arr = [4,3,2,1,0]
输出: 1
解释:
将数组分成2块或者更多块，都无法得到所需的结果。
例如，分成 [4, 3], [2, 1, 0] 的结果是 [3, 4, 0, 1, 2]，这不是有序的数组。
示例 2:

输入: arr = [1,0,2,3,4]
输出: 4
解释:
我们可以把它分成两块，例如 [1, 0], [2, 3, 4]。
然而，分成 [1, 0], [2], [3], [4] 可以得到最多的块数。

### 解题思路

注意读题！！！0到length-1的不重合数组！所以当前几个数组的最大值等于max,就证明这些可以正确排列后放在前部分，就可以加一，如果12543,1 和 2 都可以自己加一，但是到了5的时候，因为他不在他应该再的部分，导致他跟后面的43 max得出5之后，才能加一，543调整成345

### tag

logical

```python
class Solution(object):
    def maxChunksToSorted(self, arr):
        """
        :type arr: List[int]
        :rtype: int
        """
        res,m = 0,0
        for i in range(len(arr)):
            m = max(m,arr[i])
            if m==i:
                res+=1
        return res
```

## 139.单词拆分

### 题目描述

给定一个**非空**字符串 *s* 和一个包含**非空**单词列表的字典 *wordDict*，判定 *s* 是否可以被空格拆分为一个或多个在字典中出现的单词。

**说明：**

- 拆分时可以重复使用字典中的单词。
- 你可以假设字典中没有重复的单词。

**示例 1：**

```
输入: s = "leetcode", wordDict = ["leet", "code"]
输出: true
解释: 返回 true 因为 "leetcode" 可以被拆分成 "leet code"。
```

**示例 2：**

```
输入: s = "applepenapple", wordDict = ["apple", "pen"]
输出: true
解释: 返回 true 因为 "applepenapple" 可以被拆分成 "apple pen apple"。
     注意你可以重复使用字典中的单词。
```

### 解题思路

dp,dp[i]为True是因为与j之间的单词存在而且dp[i-j]为True

### tag

Dp

```python
class Solution(object):
    def wordBreak(self, s, words):
        df = [False]*(len(s)+1)
        df[0] = True
        for i in range(len(s)):
            for w in words:
                if i-len(w)+1>=0 and w==s[i-len(w)+1:i+1] and df[i-len(w)+1]:
                    df[i+1] = True
        return df[len(s)]                
```

## 781.森林中的兔子

### 题目描述

```
森林中，每个兔子都有颜色。其中一些兔子（可能是全部）告诉你还有多少其他的兔子和自己有相同的颜色。我们将这些回答放在 answers 数组里。

返回森林中兔子的最少数量。

示例:
输入: answers = [1, 1, 2]
输出: 5
解释:
两只回答了 "1" 的兔子可能有相同的颜色，设为红色。
之后回答了 "2" 的兔子不会是红色，否则他们的回答会相互矛盾。
设回答了 "2" 的兔子为蓝色。
此外，森林中还应有另外 2 只蓝色兔子的回答没有包含在数组中。
因此森林中兔子的最少数量是 5: 3 只回答的和 2 只没有回答的。

输入: answers = [10, 10, 10]
输出: 11

输入: answers = []
输出: 0
```

### 解题思路

逻辑

### tag

logical

```python
class Solution(object):
    def numRabbits(self, answers):
        """
        :type answers: List[int]
        :rtype: int
        """
        if not answers:
            return 0
        dic = {}
        for num in answers:
            if num not in dic:
                dic[num]=1
            else:
                dic[num]+=1
        sum = 0
        for key,value in dic.items():
            if key==0:
                sum+=value
            else:
                if key<value:
                    while(key<value):
                        sum += (key+1)
                        value = value-key-1
                    if value!=0:
                        sum+=(1+key)
                else:
                    sum +=(1+key)
        return sum
```

## 763.划分字母区间

### 题目描述

字符串 S 由小写字母组成。我们要把这个字符串划分为尽可能多的片段，同一个字母只会出现在其中的一个片段。返回一个表示每个字符串片段的长度的列表。

示例 1:

输入: S = "ababcbacadefegdehijhklij"
输出: [9,7,8]
解释:
划分结果为 "ababcbaca", "defegde", "hijhklij"。
每个字母最多出现在一个片段中。
像 "ababcbacadefegde", "hijhklij" 的划分是错误的，因为划分的片段数较少。

### 解题思路

贪心，从后往前先记录每个字母最后出现的位置，然后再从头到尾遍历，找到每个start，maxend，

### tag

greedy

### 解法

```python
class Solution(object):
    def partitionLabels(self, S):
        """
        :type S: str
        :rtype: List[int]
        """
        dic = {}
        for i in range(len(S),-1,-1):
            if S[i] not in dic:
                dic[S[i]] == i
        start,end = 0,0
        res = []
        for index,s in enumerate(S):
            if dic[S[index]]>end:
                end = dic[S[index]]
            if end==index:
                res.append(end-start)
                start,end = i+1,i+1
        return res
```

## 775.全局倒置和局部倒置

### 题目描述

数组 A 是 [0, 1, ..., N - 1] 的一种排列，N 是数组 A 的长度。全局倒置指的是 i,j 满足 0 <= i < j < N 并且 A[i] > A[j] ，局部倒置指的是 i 满足 0 <= i < N 并且 A[i] > A[i+1] 。

当数组 A 中全局倒置的数量等于局部倒置的数量时，返回 true 。

 

示例 1:

输入: A = [1,0,2]
输出: true
解释: 有 1 个全局倒置，和 1 个局部倒置。
示例 2:

输入: A = [1,2,0]
输出: false
解释: 有 2 个全局倒置，和 1 个局部倒置。

### 解题思路

隔一个或几个以外的前大于后则Flase

### tag

logical

### 解法

```python
class Solution(object):
    def isIdealPermutation(self, A):
        """
        :type A: List[int]
        :rtype: bool
        """
        if len(A)==1 or len(A)==2:
            return True
        m = A[0]
        for i in range(2,len(A)):
            if A[i]<m:
                return False
            else:
                m = min(m,A[i-1])
        return True
```

## 779. 第K个语法符号

在第一行我们写上一个 0。接下来的每一行，将前一行中的0替换为01，1替换为10。

给定行数 N 和序数 K，返回第 N 行中第 K个字符。（K从1开始）


例子:

输入: N = 1, K = 1
输出: 0

输入: N = 2, K = 1
输出: 0

输入: N = 2, K = 2
输出: 1

输入: N = 4, K = 5
输出: 1

解释:
第一行: 0
第二行: 01
第三行: 0110
第四行: 01101001

### 解题思路

找规律，直接递归（注意python3和python2的区别， 在2中：1/2 = 0 ,3中， 1/2 = 0.5）

### tag

递归

### 解法

```python
import math
class Solution(object):
    def kthGrammar(self, N, K):
        """
        :type N: int
        :type K: int
        :rtype: int
        """
        if N == 1:
            return 0
        else:
            order = math.ceil(K / 2.0)
            num = self.kthGrammar(N - 1, math.ceil(K / 2.0))
            if num == 0:
                if K % 2 != 0:
                    return 0
                else:
                    return 1
            else:
                if K % 2 != 0:
                    return 1
                else:
                    return 0

```

## 740.删除和获得点数

给定一个整数数组 nums ，你可以对它进行一些操作。

每次操作中，选择任意一个 nums[i] ，删除它并获得 nums[i] 的点数。之后，你必须删除每个等于 nums[i] - 1 或 nums[i] + 1 的元素。

开始你拥有 0 个点数。返回你能通过这些操作获得的最大点数。

示例 1:

输入: nums = [3, 4, 2]
输出: 6
解释: 
删除 4 来获得 4 个点数，因此 3 也被删除。
之后，删除 2 来获得 2 个点数。总共获得 6 个点数

### 解题思路

注意读题！！！每次操作=删了一个，获得这个的点数，删除所有与之临近值的元素，随后继续，排序后like house rober，简单的dp，dp[i] = max(dp[i-1],dp[i-2]+v)， key：要知道，dp[i-1]>=dp[i-2]，所以当隔一个时候，直接dp[i-2]+v.

### tag

dp

### 解法

```python
class Solution(object):
    def deleteAndEarn(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        dic = {}
        for num in nums:
            dic[num] = dic.get(num,0)+num
        res = [0]
        key_set = set()
        for k,v in sorted(dic.items()):
            if k-1 not in key_set:
                key_set.add(k)
                res.append(res[-1]+v)
            else:
                key_set.add(k)
                res.append(max(res[-2]+v,res[-1]))
        return res[-1]
```

## 783.二叉搜索树结点最小距离

给定一个二叉搜索树的根结点 root, 返回树中任意两节点的差的最小值。

示例：

输入: root = [4,2,6,1,3,null,null]
输出: 1
解释:
注意，root是树结点对象(TreeNode object)，而不是数组。

给定的树 [4,2,6,1,3,null,null] 可表示为下图:

          4
        /   \
      2      6
     / \    
    1   3  

最小的差值是 1, 它是节点1和节点2的差值, 也是节点3和节点2的差值。

### 解题思路

二叉搜索树，中序遍历.

### tag

BST

### 解法1 

```python
    pre = -float('inf')
    res = float('inf')

    def minDiffInBST(self, root):
        if root is None:
            return
        
        self.minDiffInBST(root.left)
		# evaluate and set the current node as the node previously evaluated
        self.res = min(self.res, root.val - self.pre)
        self.pre = root.val
		
        self.minDiffInBST(root.right)
        return self.res
```

### 解法2

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def minDiffInBST(self, root: TreeNode) -> int:
        res = []
        self.part(root.left,res)
        res.append(root.val)
        self.part(root.right,res)
        m = 0xfffffff
        for i in range(len(res)-1):
            m= min(m,res[i+1]-res[i])
        return m
    def part(self,root,res):
        if not root:
            return
        self.part(root.left,res)
        res.append(root.val)
        self.part(root.right,res)
```

## 788.旋转数字

### 题目描述

我们称一个数 X 为好数, 如果它的每位数字逐个地被旋转 180 度后，我们仍可以得到一个有效的，且和 X 不同的数。要求每位数字都要被旋转。

如果一个数的每位数字被旋转以后仍然还是一个数字， 则这个数是有效的。0, 1, 和 8 被旋转后仍然是它们自己；2 和 5 可以互相旋转成对方；6 和 9 同理，除了这些以外其他的数字旋转以后都不再是有效的数字。

现在我们有一个正整数 N, 计算从 1 到 N 中有多少个数 X 是好数？

示例:
输入: 10
输出: 4
解释: 
在[1, 10]中有四个好数： 2, 5, 6, 9。
注意 1 和 10 不是好数, 因为他们在旋转之后不变。

### 解题思路

dp，记录住之前的部分，1代表可以用，但是不存在旋转变换，2代表存在旋转的

### tag

dp

### 解法

```python
class Solution:
    def rotatedDigits(self, N: int) -> int:
        dp = [0]*(N+1)
        count = 0
        for i in range(N+1):
            if i<10:
                if i==0 or i==1 or i==8:
                    dp[i]=1
                elif i==2 or i==5 or i==6 or i==9:
                    dp[i]=2
                    count+=1
            else:
                a,b = dp[i//10],dp[i%10]
                if a==1 and b==1 :
                    dp[i]=1
                elif a>=1 and b>=1:
                    dp[i] =2 
                    count+=1
        return count
        
```

## 801. 使序列递增的最小交换次数

### 题目描述

我们有两个长度相等且不为空的整型数组 A 和 B 。

我们可以交换 A[i] 和 B[i] 的元素。注意这两个元素在各自的序列中应该处于相同的位置。

在交换过一些元素之后，数组 A 和 B 都应该是严格递增的（数组严格递增的条件仅为A[0] < A[1] < A[2] < ... < A[A.length - 1]）。

给定数组 A 和 B ，请返回使得两个数组均保持严格递增状态的最小交换次数。假设给定的输入总是有效的。

示例:
输入: A = [1,3,5,4], B = [1,2,3,7]
输出: 1
解释: 
交换 A[3] 和 B[3] 后，两个数组如下:
A = [1, 3, 5, 7] ， B = [1, 2, 3, 4]
两个数组均为严格递增的。

### 解题思路

dp，从左往右，记录swap[i],not_swap[i],然后根据这个去一步步判断

### tag

dp

### 解法

```python
class Solution:
    def minSwap(self, A: List[int], B: List[int]) -> int:
        swap,not_swap = [len(A)]*len(A),[len(A)]*len(A)
        swap[0],not_swap[0] = 1,0
        for i in range(1,len(A)):
            if A[i]>A[i-1] and B[i]>B[i-1]:
                not_swap[i] = not_swap[i-1]
                swap[i] = swap[i-1]+1
            if A[i]>B[i-1] and B[i]>A[i-1]:
                swap[i] = min(swap[i],not_swap[i-1]+1)
                not_swap[i] = min(not_swap[i],swap[i-1])
        return min(not_swap[-1],swap[-1])
```

##  807. 保持城市天际线

### 题目描述

在二维数组grid中，grid[i][j]代表位于某处的建筑物的高度。 我们被允许增加任何数量（不同建筑物的数量可能不同）的建筑物的高度。 高度 0 也被认为是建筑物。

最后，从新数组的所有四个方向（即顶部，底部，左侧和右侧）观看的“天际线”必须与原始数组的天际线相同。 城市的天际线是从远处观看时，由所有建筑物形成的矩形的外部轮廓。 请看下面的例子。

建筑物高度可以增加的最大总和是多少？

例子：
输入： grid = [[3,0,8,4],[2,4,5,7],[9,2,6,3],[0,3,1,0]]
输出： 35
解释： 
The grid is:
[ [3, 0, 8, 4], 
  [2, 4, 5, 7],
  [9, 2, 6, 3],
  [0, 3, 1, 0] ]

从数组竖直方向（即顶部，底部）看“天际线”是：[9, 4, 8, 7]
从水平水平方向（即左侧，右侧）看“天际线”是：[8, 7, 9, 3]

在不影响天际线的情况下对建筑物进行增高后，新数组如下：

gridNew = [ [8, 4, 8, 7],
            [7, 4, 7, 7],
            [9, 4, 8, 7],
            [3, 3, 3, 3] ]

### 解题思路

逻辑

### tag

logical

### 解法

```python
class Solution(object):
    def maxIncreaseKeepingSkyline(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        row = len(grid)
        col = len(grid[0])
        a = [0]*col
        b = [0]*row
        count = 0
        for j in range(col):
            m = -0xfffffff
            for i in range(row):
                m = max(m,grid[i][j])
            a[j] = m
        for i in range(row):
            m = -0xfffffff
            for j in range(col):
                m = max(m,grid[i][j])
            b[i] = m     
        for i in range(row):
            for j in range(col):
                temp = min(b[i],a[j])
                count = count+(temp-grid[i][j])
        return count
     def maxIncreaseKeepingSkyline(self, grid):
        row, col = map(max, grid), map(max, zip(*grid))
        return sum(min(i, j) for i in row for j in col) - sum(map(sum, grid))           
```

## 856. 括号的分数

### 题目描述

给定一个平衡括号字符串 S，按下述规则计算该字符串的分数：

() 得 1 分。
AB 得 A + B 分，其中 A 和 B 是平衡括号字符串。
(A) 得 2 * A 分，其中 A 是平衡括号字符串。


示例 1：

输入： "()"
输出： 1
示例 2：

输入： "(())"
输出： 2

### 解题思路

用堆，有0碰到右括号，说明他是1，没0的话乘以2

逻辑

### tag

stack

### 解法

```python
class Solution(object):
    def maxIncreaseKeepingSkyline(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        row = len(grid)
        col = len(grid[0])
        a = [0]*col
        b = [0]*row
        count = 0
        for j in range(col):
            m = -0xfffffff
            for i in range(row):
                m = max(m,grid[i][j])
            a[j] = m
        for i in range(row):
            m = -0xfffffff
            for j in range(col):
                m = max(m,grid[i][j])
            b[i] = m     
        for i in range(row):
            for j in range(col):
                temp = min(b[i],a[j])
                count = count+(temp-grid[i][j])
        return count
     def maxIncreaseKeepingSkyline(self, grid):
        row, col = map(max, grid), map(max, zip(*grid))
        return sum(min(i, j) for i in row for j in col) - sum(map(sum, grid))           
```

## 861. 翻转矩阵后的得分

### 题目描述

有一个二维矩阵 A 其中每个元素的值为 0 或 1 。

移动是指选择任一行或列，并转换该行或列中的每一个值：将所有 0 都更改为 1，将所有 1 都更改为 0。

在做出任意次数的移动后，将该矩阵的每一行都按照二进制数来解释，矩阵的得分就是这些数字的总和。

返回尽可能高的分数。

 

示例：

输入：[[0,0,1,1],[1,0,1,0],[1,1,0,0]]
输出：39
解释：
转换为 [[1,1,1,1],[1,0,0,1],[1,1,1,1]]
0b1111 + 0b1001 + 0b1111 = 15 + 9 + 15 = 39

### 解题思路

贪婪算法，第一个必须得等于2，后面的话就只能列变换

逻辑

### tag

greedy

### 解法

```python
class Solution:
    def matrixScore(self, A: List[List[int]]) -> int:
        a,b = len(A),len(A[0])
        res = 0
        lis = [0]*b
        for i in range(a):
            if A[i][0]==0:
                for j in range(1,b):
                    if A[i][j]==0:
                        lis[j]+=1
            else:
                for j in range(1,b):
                    if A[i][j]==1:
                        lis[j]+=1                
        res+=a*(2**(b-1))
        for i in range(1,len(lis)):
            res+=max(lis[i],a-lis[i])*(2**(b-1-i))
        return res
        def matrixScore(self, A):
            M, N = len(A), len(A[0])
            res = (1 << N - 1) * M
            for j in range(1, N):
                cur = sum(A[i][j] == A[i][0] for i in range(M))
                res += max(cur, M - cur) * (1 << N - 1 - j)
            return res
```

## 863. 二叉树中所有距离为 K 的结点

### 题目描述

给定一个二叉树（具有根结点 root）， 一个目标结点 target ，和一个整数值 K 。

返回到目标结点 target 距离为 K 的所有结点的值的列表。 答案可以以任何顺序返回。

输入：root = [3,5,1,6,2,0,8,null,null,7,4], target = 5, K = 2

输出：[7,4,1]

解释：
所求结点为与目标结点（值为 5）距离为 2 的结点，
值分别为 7，4，以及 1

### 解题思路

先dfs，把节点相邻的集合放入map中，随后bfs，层次遍历

### tag

bfs,dfs

### 解法

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
import collections
class Solution:
    def distanceK(self, root, target, K):
        """
        :type root: TreeNode
        :type target: TreeNode
        :type K: int
        :rtype: List[int]
        """
        conn = collections.defaultdict(list)
        def connect(par,child):
            if  par and  child:
                conn[par.val].append(child.val)
                conn[child.val].append(par.val)
            if child.left:connect(child,child.left)
            if child.right:connect(child,child.right)
        connect(None,root)
        bfs = [target.val]
        seen = set(bfs)
        for i in range(K):
            bfs = [y for x in bfs for y in conn[x] if y not in seen]
            seen |= set(bfs)
        return bfs
```

## 865. 具有所有最深结点的最小子树

### 题目描述

给定一个根为 root 的二叉树，每个结点的深度是它到根的最短距离。

如果一个结点在整个树的任意结点之间具有最大的深度，则该结点是最深的。

一个结点的子树是该结点加上它的所有后代的集合。

返回能满足“以该结点为根的子树中包含所有最深的结点”这一条件的具有最大深度的结点。

### 解题思路

递归，每次分左右，然后返回的两部分，一部分是当前的深度，一部分是当前部分的子树答案

如果l[0]>r[0]，则说明在左边，反之在右边

### tag

递归

### 解法

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def subtreeWithAllDeepest(self, root: TreeNode) -> TreeNode:
        def deep(root):
            if not root:return 0,None            
            l,r = deep(root.left),deep(root.right)
            if l[0]>r[0]:return l[0]+1,l[1]
            elif r[0]>l[0]:return r[0]+1,r[1]
            else:return l[0]+1,root
        return deep(root)[1]
```

## 868. 二进制间距

### 题目描述

给定一个正整数 `N`，找到并返回 `N` 的二进制表示中两个连续的 1 之间的最长距离。 

如果没有两个连续的 1，返回 `0` 。

### 解题思路

用bin()

如果l[0]>r[0]，则说明在左边，反之在右边

### tag

easy

### 解法

```python
class Solution:
    def binaryGap(self, N: int) -> int:
        pre,dis = 0,0
        for i,num in enumerate(bin(N)[2:]):
            if num == "1":
                dis = max(dis,i-pre)
                pre = i
        return dis
```

## 869. 重新排序得到 2 的幂

### 题目描述

给定正整数 N ，我们按任何顺序（包括原始顺序）将数字重新排序，注意其前导数字不能为零。

如果我们可以通过上述方式得到 2 的幂，返回 true；否则，返回 false。

示例 1：

输入：1
输出：true
示例 2：

输入：10
输出：false

### 解题思路

Counter,遍历

### tag

Counter

### 解法

```python
from collections import Counter
class Solution:
    def reorderedPowerOf2(self, N: int) -> bool:
        b = Counter(str(N))
        lb = len(str(N))
        power = 1
        while(True):
            la = len(str(power))
            if la<lb:
                power = power*2
                continue
            elif la>lb:
                break
            else:
                if Counter(str(power)) == b:
                    return True
                else:
                    power = power*2
                    continue
        return False
```

## 2. 两数字相加

### 题目描述

给出两个 非空 的链表用来表示两个非负的整数。其中，它们各自的位数是按照 逆序 的方式存储的，并且它们的每个节点只能存储 一位 数字。

如果，我们将这两个数相加起来，则会返回一个新的链表来表示它们的和。

您可以假设除了数字 0 之外，这两个数都不会以 0 开头。

示例：

输入：(2 -> 4 -> 3) + (5 -> 6 -> 4)
输出：7 -> 0 -> 8
原因：342 + 465 = 807

### 解题思路



### tag



### 解法

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        carry = 0
        root = n = ListNode(0)
        while l1 or l2 or carry:
            v1,v2 = 0,0
            if l1:
                v1 = l1.val
                l1 = l1.next
            if l2:
                v2 = l2.val
                l2 = l2.next
            carry,v = divmod(v1+v2+carry,10)
            node = ListNode(v)
            n.next = node
            n = n.next
        return root.next
```

## 11. 盛水最多的容器

### 题目描述

给定 n 个非负整数 a1，a2，...，an，每个数代表坐标中的一个点 (i, ai) 。在坐标内画 n 条垂直线，垂直线 i 的两个端点分别为 (i, ai) 和 (i, 0)。找出其中的两条线，使得它们与 x 轴共同构成的容器可以容纳最多的水。

说明：你不能倾斜容器，且 n 的值至少为 2。

### 解题思路

看最小的那边，如果向里收缩比他还小，就继续收缩，出现一个大一些的值，比较

### tag



### 解法

```python
class Solution:
    def maxArea(self, height: List[int]) -> int:
        l,r = 0,len(height)-1
        m = -1
        while(l<r):
            a,b = height[l],height[r]
            if a<b:
                area = a*(r-l)
                while(height[l]<=a):
                    l+=1
            else:
                area = b*(r-l)
                while(height[r]<=b) and r:
                    r-=1
            m = max(m,area)
        return m
```

## 15. three sum

### 题目描述

给定一个包含 n 个整数的数组 nums，判断 nums 中是否存在三个元素 a，b，c ，使得 a + b + c = 0 ？找出所有满足条件且不重复的三元组。

注意：答案中不可以包含重复的三元组。

例如, 给定数组 nums = [-1, 0, 1, 2, -1, -4]，

满足要求的三元组集合为：
[
  [-1, 0, 1],
  [-1, -1, 2]
]

### 解题思路

从头到l-2开始遍历，右边双指针，移动

### tag



### 解法

```python
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        res = []
        for i in range(len(nums)-2):
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            a,b = i+1,len(nums)-1
            temp = -nums[i]
            while(b>a):
                if temp==(nums[b]+nums[a]):
                    # t_res = [nums[i],nums[a],nums[b]]
                    # if t_res not in res:
                    #     res.append(t_res)
                    # a+=1
                    while a < len(nums) - 1 and nums[a] == nums[a + 1]:
                        a += 1
                    while b > 0 and nums[b] == nums[b - 1]:
                        b -= 1
                    res.append([nums[i], nums[a], nums[b]])
                    a += 1
                elif temp>(nums[b]+nums[a]):
                    a+=1
                else:
                    b-=1
        return res
```

## 55555.  拓扑排序 

### 题目描述

拓扑排序

A mapping of digit to letters (just like on the telephone buttons) is given below. Note that 1 does not map to any letters.

**Example:**

```
Input: G = {
    'a':'bf',
    'b':'cdf',
    'c':'d',
    'd':'ef',
    'e':'f',
    'f':''
}
Output: ['a', 'b', 'c', 'd', 'e', 'f']
```

### 解题思路

统计每个节点的入度，入度为0的形成一个栈，每次出一个入度为0的数，以他为起点的节点入度减一。

### tag

graph

### 解法

```python
class Solution(object):
    def letterCombinations(self, digits):
        """
        :type digits: str
        :rtype: List[str]
        """
        if not digits:
            return []
        lis = [["a", "b", "c"], ["d", "e", "f"], ["g", "h", "i"], ["j", "k", "l"], ["m", "n", "o"],
               ["p", "q", "r", 's'], ["t", "u", "v"], ["w", "x", "y", "z"]]
        res = []
        part = ""
        self.dfs(digits, lis, part, res)
        return res

    def dfs(self, nums, lis, part, res):
        if not nums:
            res.append(part)
        else:
            num = int(nums[0]) - 2
            for let in lis[num]:
                self.dfs(nums[1:], lis, part + let, res)
```

## 207.  Course Schedule I

### 题目描述

现在你总共有 n 门课需要选，记为 0 到 n-1。

在选修某些课程之前需要一些先修课程。 例如，想要学习课程 0 ，你需要先完成课程 1 ，我们用一个匹配来表示他们: [0,1]

给定课程总量以及它们的先决条件，返回你为了学完所有课程所安排的学习顺序。

可能会有多个正确的顺序，你只要返回一种就可以了。如果不可能完成所有课程，返回一个空数组。

示例 1:

输入: 2, [[1,0]] 
输出: [0,1]
解释: 总共有 2 门课程。要学习课程 1，你需要先完成课程 0。因此，正确的课程顺序为 [0,1] 。

### 解题思路

graph，拓扑排序

### tag

Graph

### 解法

```python
import collections
class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        G = collections.defaultdict(list)
        dic = dict((n,0) for n in range(numCourses))
        for u,v in prerequisites:
            dic[u]+=1
            G[v].append(u)
        zero = [i for i in dic if dic[i]==0]
        if not zero:
            return False
        res = 0
        while(zero):
            p = zero.pop()
            res+=1
            for v in G[p]:
                dic[v]-=1
                if dic[v]==0:
                    zero.append(v)
        if res!=numCourses:
            return False
        return True
                
```

## 207.  Course Schedule II

### 题目描述

现在你总共有 n 门课需要选，记为 0 到 n-1。

在选修某些课程之前需要一些先修课程。 例如，想要学习课程 0 ，你需要先完成课程 1 ，我们用一个匹配来表示他们: [0,1]

给定课程总量以及它们的先决条件，返回你为了学完所有课程所安排的学习顺序。

可能会有多个正确的顺序，你只要返回一种就可以了。如果不可能完成所有课程，返回一个空数组。

示例 1:

输入: 2, [[1,0]] 
输出: [0,1]
解释: 总共有 2 门课程。要学习课程 1，你需要先完成课程 0。因此，正确的课程顺序为 [0,1] 。

### 解题思路

graph，拓扑排序

### tag

Graph

### 解法

```python
class Solution:
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        G = collections.defaultdict(list)
        dic = dict((n,0) for n in range(numCourses))
        for u,v in prerequisites:
            dic[u]+=1
            G[v].append(u)
        zero = [i for i in dic if dic[i]==0]
        if not zero:
            return []
        res = []
        while(zero):
            p = zero.pop()
            res.append(p)
            for v in G[p]:
                dic[v]-=1
                if dic[v]==0:
                    zero.append(v)
        if len(res)!=numCourses:
            return []
        return res        
```

​	

## 24.  两两交换链表中的节点

### 题目描述

给定一个链表，两两交换其中相邻的节点，并返回交换后的链表。

你不能只是单纯的改变节点内部的值，而是需要实际的进行节点交换。

 

示例:

给定 1->2->3->4, 你应该返回 2->1->4->3

### 解题思路

遍历

### tag

ListNode

### 解法

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def swapPairs(self, head: ListNode) -> ListNode:
        dummy = ListNode(0)
        dummy.next = head
        self.part(dummy)
        return dummy.next
    
    def part(self,pre):
        if pre.next and pre.next.next:
            last = pre.next.next.next
            p,q = pre.next,pre.next.next
            pre.next = q
            p.next = last
            q.next = p
            self.part(p)
        else:
            return 
    
```

## 31.  下一个排列

### 题目描述

实现获取下一个排列的函数，算法需要将给定数字序列重新排列成字典序中下一个更大的排列。

如果不存在下一个更大的排列，则将数字重新排列成最小的排列（即升序排列）。

必须原地修改，只允许使用额外常数空间。

以下是一些例子，输入位于左侧列，其相应输出位于右侧列。
1,2,3 → 1,3,2
3,2,1 → 1,2,3
1,1,5 → 1,5,1

### 解题思路

找到第一个下降的地方，和下降序列中最小的大于他的swap，随后降下降序列reverse

### tag



### 解法

```python
class Solution:
    def nextPermutation(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        i,j = len(nums)-1,len(nums)-1
        while(i>0 and nums[i]<=nums[i-1]):
            i-=1
        if i==0:
            self.re(nums,0,j)
            return
        while(nums[j]<=nums[i-1]):
            j-=1
        temp = nums[i-1]
        nums[i-1] = nums[j]
        nums[j] = temp
        self.re(nums,i,len(nums)-1)
        
    def re(self,nums,i,j):
        while(i<j):
            temp = nums[i]
            nums[i] = nums[j]
            nums[j] = temp
            i+=1
            j-=1
```

## 33.  旋转数组

### 题目描述

假设按照升序排序的数组在预先未知的某个点上进行了旋转。

( 例如，数组 [0,1,2,4,5,6,7] 可能变为 [4,5,6,7,0,1,2] )。

搜索一个给定的目标值，如果数组中存在这个目标值，则返回它的索引，否则返回 -1 。

你可以假设数组中不存在重复的元素。

你的算法时间复杂度必须是 O(log n) 级别。

示例 1:

输入: nums = [4,5,6,7,0,1,2], target = 0
输出: 4

### 解题思路

找到第一个下降的地方，和下降序列中最小的大于他的swap，随后降下降序列reverse

### tag



### 解法

```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        left,right = 0,len(nums)-1
        if not nums:
            return -1
        while(left<=right):
            mid = (left+right)//2
            if target==nums[mid]:
                return mid
            if nums[mid]>=nums[left]:
                if target<=nums[mid] and target>=nums[left]:
                    right = mid-1
                else:
                    left = mid+1
            else:
                if target>=nums[mid] and target<=nums[right]:
                    left = mid+1
                else:
                    right = mid-1     
        return -1
```

## 33.  旋转数组

### 题目描述

```
Input: candidates = [10,1,2,7,6,1,5], target = 8,
A solution set is:
[
  [1, 7],
  [1, 2, 5],
  [2, 6],
  [1, 1, 6]
]
```

### 解题思路

binary search

### tag



### 解法

```python
class Solution:
    def combinationSum2(self, candidates, target):
        # Sorting is really helpful, se we can avoid over counting easily
        candidates.sort()                      
        result = []
        self.combine_sum_2(candidates, 0, [], result, target)
        return result

    def combine_sum_2(self, nums, start, path, result, target):
        # Base case: if the sum of the path satisfies the target, we will consider 
        # it as a solution, and stop there
        if not target:
            result.append(path)
            return

        for i in range(start, len(nums)):
            # Very important here! We don't use `i > 0` because we always want 
            # to count the first element in this recursive step even if it is the same 
            # as one before. To avoid overcounting, we just ignore the duplicates
            # after the first element.
            if i > start and nums[i] == nums[i - 1]:
                continue

            # If the current element is bigger than the assigned target, there is 
            # no need to keep searching, since all the numbers are positive
            if nums[i] > target:
                break

            # We change the start to `i + 1` because one element only could
            # be used once
            self.combine_sum_2(nums, i + 1, path + [nums[i]], 
                               result, target - nums[i])
```

## 46.  全排列

### 题目描述

给定一个没有重复数字的序列，返回其所有可能的全排列。

示例:

输入: [1,2,3]
输出:
[
  [1,2,3],
  [1,3,2],
  [2,1,3],
  [2,3,1],
  [3,1,2],
  [3,2,1]
]

### 解题思路

back tracing

### tag

back tracing

### 解法

```python
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        res = []
        self.part(nums,[],res)
        return res
    
    def part(self,nums,lis,res):
        if not nums:
            res.append(lis)
            return
        for i in range(len(nums)):
            self.part(nums[:i]+nums[i+1:],lis+[nums[i]],res)
```

## 48.  旋转图片

### 题目描述

```
Given input matrix = 
[
  [1,2,3],
  [4,5,6],
  [7,8,9]
],

rotate the input matrix in-place such that it becomes:
[
  [7,4,1],
  [8,5,2],
  [9,6,3]
]
```

### 解题思路

先斜对角线翻转，然后再每行翻转

### tag



### 解法

```python
class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        n = len(matrix)
        for i in range(n):
            for j in range(i):
                matrix[i][j],matrix[j][i] = matrix[j][i],matrix[i][j]
        for row in matrix:
            for i in range(n//2):
                row[i],row[~i] = row[~i],row[i]
```



## 49.  [ 字母异位词分组

### 题目描述

```
给定一个字符串数组，将字母异位词组合在一起。字母异位词指字母相同，但排列不同的字符串。

示例:

输入: ["eat", "tea", "tan", "ate", "nat", "bat"],
输出:
[
  ["ate","eat","tea"],
  ["nat","tan"],
  ["bat"]
]

```

### 解题思路

defaultdict,第一种，用数字记录26位，另一种，sorted

### tag



### 解法

```python
import collections
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        res =[]
        dic = collections.defaultdict(list)
        for word in strs:
            num = 0
            for l in word:
                num+=10**(ord(l)-ord("a"))
            dic[num].append(word)
        for key in dic:
            res.append(dic[key])
        return res

import  collections
class Solution:
    def groupAnagrams(self, strs):

        dic = collections.defaultdict(list)
        for string in strs:
            dic[''.join(sorted(string))] += [string]

        return [value for key, value in dic.items()]
```

## 49.   字母异位词分组

### 题目描述

```
给定一个整数数组 nums ，找到一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。

示例:

输入: [-2,1,-3,4,-1,2,1,-5,4],
输出: 6
解释: 连续子数组 [4,-1,2,1] 的和最大，为 6。
```

### 解题思路

1.DP 长度为N的数组，每个位置表示以他为结尾的最大子序和

2.分治

### tag



### 解法

```python
# DP
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        N = len(nums)
        res = [0]*N
        res[0] = nums[0]
        for i in range(1,N):
            res[i] = nums[i] if res[i-1]<0 or nums[i]+res[i-1]
        return max(res)
```



```java
#分治
class Solution {
public:
    int maxSubArray(int A[], int n) {
        // IMPORTANT: Please reset any member data you declared, as
        // the same Solution instance will be reused for each test case.
        if(n==0) return 0;
        return maxSubArrayHelperFunction(A,0,n-1);
    }
    
    int maxSubArrayHelperFunction(int A[], int left, int right) {
        if(right == left) return A[left];
        int middle = (left+right)/2;
        int leftans = maxSubArrayHelperFunction(A, left, middle);
        int rightans = maxSubArrayHelperFunction(A, middle+1, right);
        int leftmax = A[middle];
        int rightmax = A[middle+1];
        int temp = 0;
        for(int i=middle;i>=left;i--) {
            temp += A[i];
            if(temp > leftmax) leftmax = temp;
        }
        temp = 0;
        for(int i=middle+1;i<=right;i++) {
            temp += A[i];
            if(temp > rightmax) rightmax = temp;
        }
        return max(max(leftans, rightans),leftmax+rightmax);
    }
};
```

## 55.   跳跃游戏

### 题目描述

```
输入: [2,3,1,1,4]
输出: true
解释: 我们可以先跳 1 步，从位置 0 到达 位置 1, 然后再从位置 1 跳 3 步到达最后一个位置。
```

### 解题思路

1.DP greedy

### tag



### 解法

```python
#超时version
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        res = [False] * len(nums)
        res[0] = True
        maxIdx = 0
        for i in range(len(nums)-1):
            if res[i]!=True:
                continue
            if maxIdx >= i + nums[i]:
                continue
            else:
                for j in range(maxIdx+1,min(i+nums[i]+1,len(nums))):
                    res[j] = True
        return res[-1]
```



```python
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        u,g,b = 0,1,2
        l = len(nums)
        res = [u]*l
        res[l-1]=g
        for i in range(l-2,-1,-1):
            for j in range(i+1,min(i+nums[i],l-1)+1):
                if res[j]==g:
                    res[i]=g
                    break
        
        return res[0]==g
```

## 56.  融合间隔

### 题目描述

```
Input: [[1,3],[2,6],[8,10],[15,18]]
Output: [[1,6],[8,10],[15,18]]
Explanation: Since intervals [1,3] and [2,6] overlaps, merge them into [1,6].
```

### 解题思路

sorted之后，遍历

### tag



### 解法

```python
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        if not intervals:
            return []
        lis = sorted(intervals)
        res = [lis[0]]
        l,r = lis[0][0],lis[0][1]
        for a,b in lis[1:]:
            if a>r:
                res.append([a,b])
                l,r = a,b
            elif  b>r:
                res[-1] = [l,b]
                r = b
            else:
                continue
        return res
```

## 62.  唯一路径

### 题目描述

```
Input: m = 3, n = 2
Output: 3
Explanation:
From the top-left corner, there are a total of 3 ways to reach the bottom-right corner:
1. Right -> Right -> Down
2. Right -> Down -> Right
3. Down -> Right -> Right
```

### 解题思路

DP，每个点等于上加下

### tag



### 解法

```python
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        p = [[0 for _ in range(n)] for _ in range(m)]
        for i in range(n):
            p[0][i]=1
        for i in range(m):
            p[i][0]=1
        for i in range(1,m):
            for j in range(1,n):
                p[i][j] = p[i-1][j]+p[i][j-1]
        return p[m-1][n-1]
```

## 72.  编辑距离

### 题目描述

```
Given two words word1 and word2, find the minimum number of operations required to convert word1 to word2.

You have the following 3 operations permitted on a word:

Insert a character
Delete a character
Replace a character
```

### 解题思路

当我们获得 D[i-1][j]，D[i][j-1] 和 D[i-1][j-1] 的值之后就可以计算出 D[i][j]。

每次只可以往单个或者两个字符串中插入一个字符

那么递推公式很显然了

如果两个子串的最后一个字母相同，word1[i] = word2[i] 的情况下：

D[i][j] = 1 + \min(D[i - 1][j], D[i][j - 1], D[i - 1][j - 1] - 1)
D[i][j]=1+min(D[i−1][j],D[i][j−1],D[i−1][j−1]−1)

否则，word1[i] != word2[i] 我们将考虑替换最后一个字符使得他们相同：

D[i][j] = 1 + \min(D[i - 1][j], D[i][j - 1], D[i - 1][j - 1])
D[i][j]=1+min(D[i−1][j],D[i][j−1],D[i−1][j−1])

### tag

DP

### 解法

```python
class Solution:
    def minDistance(self, word1, word2):
        """
        :type word1: str
        :type word2: str
        :rtype: int
        """
        n = len(word1)
        m = len(word2)
        
        # if one of the strings is empty
        if n * m == 0:
            return n + m
        
        # array to store the convertion history
        d = [ [0] * (m + 1) for _ in range(n + 1)]
        
        # init boundaries
        for i in range(n + 1):
            d[i][0] = i
        for j in range(m + 1):
            d[0][j] = j
        
        # DP compute 
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                left = d[i - 1][j] + 1
                down = d[i][j - 1] + 1
                left_down = d[i - 1][j - 1] 
                if word1[i - 1] != word2[j - 1]:
                    left_down += 1
                d[i][j] = min(left, down, left_down)
        
        return d[n][m]

```

## 79.  词语搜索

### 题目描述

```
board =
[
  ['A','B','C','E'],
  ['S','F','C','S'],
  ['A','D','E','E']
]

Given word = "ABCCED", return true.
Given word = "SEE", return true.
Given word = "ABCB", return false.
```

### 解题思路

DFS：**重要的一点 ** 在每次dfs这个点时候，记录下这个点的值，然后标记成#记录搜索过，然后都dfs四个方向后，将值重新填回！！

每次只可以往单个或者两个字符串中插入一个字符

那么递推公式很显然了

如果两个子串的最后一个字母相同，word1[i] = word2[i] 的情况下：

D[i][j] = 1 + \min(D[i - 1][j], D[i][j - 1], D[i - 1][j - 1] - 1)
D[i][j]=1+min(D[i−1][j],D[i][j−1],D[i−1][j−1]−1)

否则，word1[i] != word2[i] 我们将考虑替换最后一个字符使得他们相同：

D[i][j] = 1 + \min(D[i - 1][j], D[i][j - 1], D[i - 1][j - 1])
D[i][j]=1+min(D[i−1][j],D[i][j−1],D[i−1][j−1])

### tag

DFS

### 解法

```python
def exist(self, board, word):
    if not board:
        return False
    for i in xrange(len(board)):
        for j in xrange(len(board[0])):
            if self.dfs(board, i, j, word):
                return True
    return False

# check whether can find word, start at (i,j) position    
def dfs(self, board, i, j, word):
    if len(word) == 0: # all the characters are checked
        return True
    if i<0 or i>=len(board) or j<0 or j>=len(board[0]) or word[0]!=board[i][j]:
        return False
    tmp = board[i][j]  # first character is found, check the remaining part
    board[i][j] = "#"  # avoid visit agian 
    # check whether can find "word" along one direction
    res = self.dfs(board, i+1, j, word[1:]) or self.dfs(board, i-1, j, word[1:]) \
    or self.dfs(board, i, j+1, word[1:]) or self.dfs(board, i, j-1, word[1:])
    board[i][j] = tmp
    return res
```



## 图

### 133.克隆图

#### 解题思路：

dic存储旧点与新点的关系，stack存储还未处理完关系的点，如果该点于dic中出现过，就证明做了初始化，直接取即可，否则要重新建立新的点，并且dic添加对应关系，与每个邻居要建立关系。

```python
"""
# Definition for a Node.
class Node(object):
    def __init__(self, val, neighbors):
        self.val = val
        self.neighbors = neighbors
"""
class Solution(object):
    def cloneGraph(self, node):
        """
        :type node: Node
        :rtype: Node
        """
        if not node:
            return 
        n_copy = Node(node.val,[])
        dic = {}
        dic[node] = n_copy
        stack = [node]
        while(stack):
            v = stack.pop()
            for neighbor in v.neighbors:
                if neighbor not in dic:
                    n_copy = Node(neighbor.val,[])
                    dic[neighbor] = n_copy
                    dic[v].neighbors.append(n_copy)
                    stack.append(neighbor)
                else:
                    dic[v].neighbors.append(dic[neighbor])
        return dic[node]
```

BFS version

用queue，先进先出，层次遍历

```python
def cloneGraph1(self, node):
    if not node:
        return 
    nodeCopy = UndirectedGraphNode(node.label)
    dic = {node: nodeCopy}
    queue = collections.deque([node])
    while queue:
        node = queue.popleft()
        for neighbor in node.neighbors:
            if neighbor not in dic: # neighbor is not visited
                neighborCopy = UndirectedGraphNode(neighbor.label)
                dic[neighbor] = neighborCopy
                dic[node].neighbors.append(neighborCopy)
                queue.append(neighbor)
            else:
                dic[node].neighbors.append(dic[neighbor])
    return nodeCopy
```

### 310.最小高度树

#### 算法思路

从外向里，向洋葱一样，去掉所有叶子节点，留下小于等于2的就是目标。

```python
class Solution(object):
    def findMinHeightTrees(self, n, edges):
        """
        :type n: int
        :type edges: List[List[int]]
        :rtype: List[int]
        """
        if n==1:
            return [0]
        adj = [set() for _ in range(n)]
        for i,j in edges:
            adj[i].add(j)
            adj[j].add(i)
        leaves = [i for i in range(n) if len(adj[i])==1]
        while(n>2):
            n-=len(leaves)
            new_leaves = []
            for v in leaves:
                j = adj[v].pop()
                adj[j].remove(v)
                if len(adj[j])==1:
                    new_leaves.append(j)
            leaves = new_leaves
        return leaves
```

## 96.  唯一的二叉搜索树

### 题目描述

```
给定一个整数数组 nums ，找到一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。

示例:

输入: [-2,1,-3,4,-1,2,1,-5,4],
输出: 6
解释: 连续子数组 [4,-1,2,1] 的和最大，为 6。
```

### 解题思路

给出n个数字，能够构建出多少个不同的bst。
这道题可以用动态规划来做。那么动态规划重要的是找出状态，以及状态转移方程。
我们来考虑一下状态以及转移，我们可以列举出数组里面每一个数字i来当bst的root，然后根据bst的性质我们知道root左边的都是比他小的，右边都是比他大的。显然，左子树是用[1,i-1],右子树是用[i+1,n]构建的，那么我们会发现如果对于每一个i我们都这样的去构建这个问题就会转移到左右子树的构建上去。G(n) = F(1,n)+F(2,n)+...+F(n,n)

所以这个F关系式怎么转化呢？
 假设我们有[1,2,3,4,5,6]6个数，我选2为root，那么2的左边有[1],右边有[3,4,5,6]，所以[1]能构建bst的数目是G(1),而[3,4,5,6]能构建bst数目的是G(4)，为什么呢？因为我构建bst主要的是看增序数组的元素个数跟元素具体是什么，是不是从1开始的并没有什么关系.
 于是：
 F[2,6] = G(1)*G(4);
 F[i,n] = G(i-1)*G(n-i);
 所以这个G(n)就可以算了：
 G(n)  = G(0)*G(n-1)+G(1)*G(n-2)+...+G(n-1)*G(0)。

### tag

DP

### 解法	

 ```python
class Solution:
    def numTrees(self, n):
        res = [0] * (n+1)
        res[0] = 1
        for i in range(1, n+1):
            for j in range(i):
                res[i] += res[j] * res[i-1-j]
        return res[n]
 ```

## 98. 鉴别二叉搜索树

### 题目描述

```
    2
   / \
  1   3

Input: [2,1,3]
Output: true
```

### 解题思路

将边界条件一级一级传下来，每个点成立的条件是他自己满足，他下面的左右也满足。

### tag

DFS

### 解法	

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def isValidBST(self, root: TreeNode) -> bool:
        return self.part(root,-float("inf"),float("inf"))
    
    def part(self,root,l,r):
        if not root:
            return True
        if root.val<=l or root.val>=r:
            return False
        left = self.part(root.left,l,root.val)
        right = self.part(root.right,root.val,r)
        return left and right
```

## 121. Best Time to Buy and Sell Stock

### 题目描述

```
Input: [7,1,5,3,6,4]
Output: 5
Explanation: Buy on day 2 (price = 1) and sell on day 5 (price = 6), profit = 6-1 = 5.
             Not 7-1 = 6, as selling price needs to be larger than buying price.
```

### 解题思路

DP

当天最大利润=max（前一天最大利润，今日price - 之前最小）

### tag

DP

```
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        if len(prices)<=1:
            return 0
        mi,ma = prices[0],0
        for i in range(1,len(prices)):
            mi = min(mi,prices[i])
            ma = max(ma,prices[i]-mi)
        return ma
```

## 101. 对称树

### 题目描述

```
    1
   / \
  2   2
 / \ / \
3  4 4  3
```

### 解题思路



### tag

DP

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
from collections import deque

class Solution:
    def isSymmetric(self, root: TreeNode) -> bool:
        if not root:
            return True
        q = deque([root.left,root.right])
        while(q):
            l,r = q.popleft(),q.popleft()
            if not l and not r:
                continue
            if not l or not r or l.val!=r.val:
                return False
            q+=[l.left,r.right,l.right,r.left]
        return True

```

## 128. Longest Consecutive Sequence

### 题目描述

```
Input: [100, 4, 200, 1, 3, 2]
Output: 4
Explanation: The longest consecutive elements sequence is [1, 2, 3, 4]. Therefore its length is 4.
```

### 解题思路

放到哈希里，然后找每个序列的开始，然后计数。

### tag

hash

```python
class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        lis = set(nums)
        best = 0
        for x in lis:
            if x-1 not in lis:
                y = x+1
                while y in lis:
                    y=y+1
                best = max(best,y-x)
        return best
```

## 139. word break

### 题目描述

```
Input: s = "leetcode", wordDict = ["leet", "code"]
Output: true
Explanation: Return true because "leetcode" can be segmented as "leet code".
```

### 解题思路

DP,记录每一个点，以他结尾，是否能组成。

### tag

DP

```python
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        res= [False]*len(s)
        for i in range(len(s)):
            for w in wordDict:
                if s[i-len(w)+1:i+1]==w and (res[i-len(w)] or i-len(w)==-1):
                    res[i] = True
        return res[-1]
                
```

## 141. Linked List Cycle

### 题目描述

```
Input: head = [3,2,0,-4], pos = 1
Output: true
Explanation: There is a cycle in the linked list, where tail connects to the second node.
```

### 解题思路

slow,fast，如果成环，则总会相遇，反之如果fast或者fast.next不存在，那肯定没有环

### tag



```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def hasCycle(self, head: ListNode) -> bool:
        slow,fast = head,head
        while(fast and fast.next):
            slow = slow.next
            fast = fast.next.next
            if slow==fast:
                return True
        return False
        
```

## 146. LRU Cache

### 题目描述

```
LRUCache cache = new LRUCache( 2 /* 缓存容量 */ );

cache.put(1, 1);
cache.put(2, 2);
cache.get(1);       // 返回  1
cache.put(3, 3);    // 该操作会使得密钥 2 作废
cache.get(2);       // 返回 -1 (未找到)
cache.put(4, 4);    // 该操作会使得密钥 1 作废
cache.get(1);       // 返回 -1 (未找到)
cache.get(3);       // 返回  3
cache.get(4);       // 返回  4

```

### 解题思路

OrderedDict,move_to_end维护最新的顺序,popitem去掉最少用的

### tag

OrderedDict

```python
from collections import OrderedDict
class LRUCache(OrderedDict):

    def __init__(self, capacity: int):
        self.capacity = capacity

    def get(self, key: int) -> int:
        if key not in self:
            return -1
        self.move_to_end(key)
        return self[key]

    def put(self, key: int, value: int) -> None:
        if key in self:
            self.move_to_end(key)
        self[key] = value
        if len(self)>self.capacity:
            self.popitem(last=False)


# Your LRUCache object will be instantiated and called as such:
# obj = LRUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)
```

## 152. Maximum Product Subarray

### 题目描述

```
Input: [2,3,-2,4]
Output: 6
Explanation: [2,3] has the largest product 6.
```

### 解题思路

维护之前最大，最小，res

### tag



```python
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        imin,imax = 1,1
        res = -0xfffffff
        for num in (nums):
            if num<0:
                temp = imax
                imax = imin
                imin = temp
            imax = max(num*imax,num)
            imin = min(num*imin,num)
            res = max(res,imax)
        return res
```

## 225. Implement Stack using Queues

### 题目描述

```
MyStack stack = new MyStack();

stack.push(1);
stack.push(2);  
stack.top();   // returns 2
stack.pop();   // returns 2
stack.empty(); // returns false
```

### 解题思路

push时候，每次popleft之前所有的，到最新的后面，相当于反向的stack

### tag

stack，deque

```python
from collections import deque
class MyStack:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.q = deque()

    def push(self, x: int) -> None:
        """
        Push element x onto stack.
        """
        self.q.append(x)
        for _ in range(len(self.q)-1):
            self.q.append(self.q.popleft())
        
        
    def pop(self) -> int:
        """
        Removes the element on top of the stack and returns that element.
        """
        return self.q.popleft()

    def top(self) -> int:
        """
        Get the top element.
        """
        return self.q[0]

    def empty(self) -> bool:
        """
        Returns whether the stack is empty.
        """
        return len(self.q)==0


# Your MyStack object will be instantiated and called as such:
# obj = MyStack()
# obj.push(x)
# param_2 = obj.pop()
# param_3 = obj.top()
# param_4 = obj.empty()
```

## 155. Min Stack

### 题目描述

```
MinStack minStack = new MinStack();
minStack.push(-2);
minStack.push(0);
minStack.push(-3);
minStack.getMin();   --> Returns -3.
minStack.pop();
minStack.top();      --> Returns 0.
minStack.getMin();   --> Returns -2.
```

### 解题思路

辅助数组，用来记录最小值

### tag

stack

### code

```python
class MinStack(object):

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.data = []
        self.fuzhu = []

    def push(self, x):
        """
        :type x: int
        :rtype: None
        """
        self.data.append(x)
        if self.fuzhu and x<self.fuzhu[-1]:
            self.fuzhu.append(x)
        elif self.fuzhu:
            self.fuzhu.append(self.fuzhu[-1])
        else:
            self.fuzhu.append(x)

    def pop(self):
        """
        :rtype: None
        """
        self.fuzhu.pop()
        return self.data.pop()

    def top(self):
        """
        :rtype: int
        """
        return self.data[-1]

    def getMin(self):
        """
        :rtype: int
        """
        return self.fuzhu[-1]

# solution2
class MinStack(object):

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.data = []
        self.fuzhu = []

    def push(self, x):
        """
        :type x: int
        :rtype: None
        """
        self.data.append(x)
        if self.fuzhu and x<=self.fuzhu[-1]:
            self.fuzhu.append(x)
        elif self.fuzhu:
            return
        else:
            self.fuzhu.append(x)

    def pop(self):
        """
        :rtype: None
        """
        if self.fuzhu[-1]==self.data[-1]:
            self.fuzhu.pop()
        return self.data.pop()

    def top(self):
        """
        :rtype: int
        """
        return self.data[-1]

    def getMin(self):
        """
        :rtype: int
        """
        return self.fuzhu[-1]
```

## 208. Implement Trie

### 题目描述

```
MinStack minStack = new MinStack();
minStack.push(-2);
minStack.push(0);
minStack.push(-3);
minStack.getMin();   --> Returns -3.
minStack.pop();
minStack.top();      --> Returns 0.
minStack.getMin();   --> Returns -2.
```

### 解题思路

用字典记录每个节点的儿子

### tag



### code

```python
import collections
class TreeNode(object):
    def __init__(self):
        self.endOfWord = False
        self.children = collections.defaultdict(TreeNode)


class Trie(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.root = TreeNode()

    def insert(self, word):
        """
        Inserts a word into the trie.
        :type word: str
        :rtype: None
        """
        p = self.root
        for c in word:
            p = p.children[c]
        p.endOfWord = True

    def search(self, word):
        """
        Returns if the word is in the trie.
        :type word: str
        :rtype: bool
        """
        p = self.root
        for c in word:
            if c not in p.children:
                return False
            p = p.children[c]
        return p.endOfWord

    def startsWith(self, prefix):
        """
        Returns if there is any word in the trie that starts with the given prefix.
        :type prefix: str
        :rtype: bool
        """
        p = self.root
        for c in prefix:
            if c not in p.children:
                return False
            p = p.children[c]
        return True

    # Your Trie object will be instantiated and called as such:
# obj = Trie()
# obj.insert(word)
# param_2 = obj.search(word)
# param_3 = obj.startsWith(prefix)
```

## 208. Implement Trie

### 题目描述

```
给定一个由 '1'（陆地）和 '0'（水）组成的的二维网格，计算岛屿的数量。一个岛被水包围，并且它是通过水平方向或垂直方向上相邻的陆地连接而成的。你可以假设网格的四个边均被水包围。

示例 1:

输入:
11110
11010
11000
00000

输出: 1
```

### 解题思路

每part置零之后计数加一

### tag

DFS,BFS

### code

```python
### DFS version
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        if not grid:return 0
        a,b = len(grid),len(grid[0])
        cnt = 0
        for i in range(a):
            for j in range(b):
                if grid[i][j]=="1":
                    self.dfs(grid,i,j)
                    cnt += 1
        return cnt
    def dfs(self, grid, i , j):
        if i < 0 or i >= len(grid) or j < 0 or j >= len(grid[i]) or grid[i][j] == '0':
            return 0 
        
        grid[i][j] = '0'
        
        self.dfs(grid, i+1, j) #down
        self.dfs(grid, i-1 , j) #up
        self.dfs(grid, i, j+1) #right
        self.dfs(grid, i, j-1) #left
    
        return 1
```



## 279. Perfect Squares

### 题目描述

```
Perfect Squares
Input: n = 12
Output: 3 
Explanation: 12 = 4 + 4 + 4.

Input: n = 13
Output: 2
Explanation: 13 = 4 + 9.
```

### 解题思路

每part置零之后计数加一

### tag

DFS,BFS

### code

```python
class Solution:
    import math
    def numSquares(self, n: int) -> int:
        dp = [0]*(n+1)
        for i in range(1,n+1):
            val = 0xfffffff
            for j in range(1,int(math.sqrt(i))+1):
                val = min(val,dp[i-j*j]+1)
            dp[i]=val
        return dp[n]
    
import math
class Solution(object):
    def numSquares(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n < 2:
            return n
        lst = [i**2 for i in range(1,int(math.sqrt(n)+1))]
        cnt = 0
        toCheck = {n}
        while toCheck:
            cnt += 1
            temp = set()
            for x in toCheck:
                for y in lst:
                    if x==y:
                        return cnt
                    if x<y:
                        break
                    temp.add(x-y)
            toCheck = temp
        return cnt
```

## 240. 279. Perfect Squares

### 题目描述

```
Write an efficient algorithm that searches for a value in an m x n matrix. This matrix has the following properties:

Integers in each row are sorted in ascending from left to right.
Integers in each column are sorted in ascending from top to bottom.
```

### 解题思路

从右上角开始，往下或者往左，每次排除一行或者一列，O(m+n)的时间复杂度

### tag

逻辑

### code

```python
class Solution:
    def searchMatrix(self, matrix, target):
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        """
        if not matrix:
            return False
        row,col = 0,len(matrix[0])-1
        while(row<len(matrix) and col>-1):
            if matrix[row][col] == target:
                return True
            elif(matrix[row][col]<target):
                row +=1
            else:
                col -=1
        return False
```

## 238. Product of Array Except Self

### 题目描述

```
Input:  [1,2,3,4]
Output: [24,12,8,6]
```

### 解题思路

先把左边的积填入，随后一个个乘右边的积。

### tag



### code

```python
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        res = []
        p = 1
        for i in range(len(nums)):
            res.append(p)
            p = p*nums[i]
        p = 1
        for i in range(len(nums)-1,-1,-1):
            res[i] = res[i] * p
            p = p*nums[i]
        return res
```



## 322. coin change

### 题目描述

```
Input: coins = [1, 2, 5], amount = 11
Output: 3 
Explanation: 11 = 5 + 5 + 1
```

### 解题思路

DP,从上到下，用数组记录已经运算过的位置。

### tag

DP

### code

```python
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        count = [0] * (amount)
        return self.part(coins,amount,count)
    
    def part(self,coins,rem,count):
        if rem < 0:
            return -1
        if rem ==0 :
            return 0
        if count[rem-1] != 0:
            return count[rem-1]
        m = float("inf")
        for coin in coins:
            res = self.part(coins,rem-coin,count)
            if res>=0 and res<m:
                m = res
        count[rem-1] = m+1 if m!=float("inf") else -1
        return count[rem-1]
    
```

## 347. Top K Frequent Elements

### 题目描述

```
Input: nums = [1,1,1,2,2,3], k = 2
Output: [1,2]
```

### 解题思路

最小堆，桶排序

### tag

重要！！！stack，bucket

### code

```python
class Solution(object):
    def topKFrequent(self, nums, k):
        hs = {}
        frq = {}
        for i in xrange(0, len(nums)):
            if nums[i] not in hs:
                hs[nums[i]] = 1
            else:
                hs[nums[i]] += 1

        for z,v in hs.iteritems():
            if v not in frq:
                frq[v] = [z]
            else:
                frq[v].append(z)
        
        arr = []
        
        for x in xrange(len(nums), 0, -1):
            if x in frq:
                
                for i in frq[x]:
                    arr.append(i)

        return [arr[x] for x in xrange(0, k)]
```

```python

class Solution(object):
    def topKFrequent(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """
        res = []
        map = {}
        data = []
        for num in (nums):
            if num not in map.keys():
                map[num] = 1
            else:
                map[num] += 1
        for key in map:
            data+=[(map[key],key)]
        self.build_max_heap(data)
        print(data)
        for i in range(k):
            num,data = self.pop_max(data)
            res.append(num[1])
        return res
    def left(self, num):
        return num * 2 + 1

    def right(self, num):
        return num * 2 + 2

    def max_heapify(self, nums, i):
        l, r = self.left(i), self.right(i)
        length = len(nums)
        if l < length and nums[i][0] < nums[l][0]:
            largest = l
        else:
            largest = i
        if r < length and nums[largest][0] < nums[r][0]:
            largest = r
        if (largest != i):
            self.swap(nums, largest, i)
            self.max_heapify(nums,largest)

    def build_max_heap(self,nums):
        for i in range((int(len(nums) - 1) <<1), -1, -1):
            self.max_heapify(nums, i)

    def swap(self, nums, i, j):
        temp = nums[i]
        nums[i] = nums[j]
        nums[j] = temp

    def pop_max(self,nums):
        max = (nums[0])
        nums[0] = nums[-1]
        nums = nums[:-1]
        self.max_heapify(nums,0)
        return max,nums
```

python默认是最小堆，改成最大堆

```python
max_heap = [(-val, key) for key, val in dic.items()]
为什么是-val？
Python里面的heapify是定义的Min-heap，在StackOverFlow里面寻找Max-heap的方法，这个答案比较符合我偷懒的风格: Link, 把Value直接设成 -Value即可。

import heapq
from collections import Counter
class Solution:
    def topKFrequent(self, nums, k):
        res = []
        dic = Counter(nums)
        max_heap = [(-val, key) for key, val in dic.items()]
        heapq.heapify(max_heap)
        for i in range(k):
            res.append(heapq.heappop(max_heap)[1])
        return res   
```



## 406. Queue Reconstruction by Height

### 题目描述

```
Input:
[[7,0], [4,4], [7,1], [5,0], [6,1], [5,2]]

Output:
[[5,0], [7,0], [5,2], [6,1], [4,4], [7,1]]
```

### 解题思路

根据两列先后排序，随后greedy

### tag

greedy

### code

```python
class Solution:
    def reconstructQueue(self, people: List[List[int]]) -> List[List[int]]:
        people.sort(key = lambda x: (-x[0], x[1]))
        output = []
        for p in people:
            output.insert(p[1], p)
        return output

```



## 739. Daily Temperatures

### 题目描述

```
Given a list of daily temperatures T, return a list such that, for each day in the input, tells you how many days you would have to wait until a warmer temperature. If there is no future day for which this is possible, put 0 instead.

For example, given the list of temperatures T = [73, 74, 75, 71, 69, 72, 76, 73], your output should be [1, 1, 4, 2, 1, 1, 0, 0].
```

### 解题思路

利用stack，存储元素，当前元素比他大，则弹出他，并且计算距离，反之添加进stack

### tag

stack

### code

```python
class Solution(object):
    def dailyTemperatures(self, T):
        """
        :type T: List[int]
        :rtype: List[int]
        """
        ans = [0] * len(T)
        stack = []
        for i, t in enumerate(T):
            while stack and T[stack[-1]] < t:
                cur = stack.pop()
                ans[cur] = i - cur
            stack.append(i)

        return ans

        
```

## 647. Palindromic Substrings

### 题目描述

```
Given a string, your task is to count how many palindromic substrings in this string.

The substrings with different start indexes or end indexes are counted as different substrings even they consist of same characters.

Example 1:
Input: "abc"
Output: 3
Explanation: Three palindromic strings: "a", "b", "c".
```

### 解题思路

遍历字符串，每个位置为中心，分别为奇偶数长度，去向两侧延展。

### tag



### code

```python
class Solution:
    def countSubstrings(self, s: str) -> int:
        if not s or len(s) == 0:
            return 0
        cnt = 0
        for i in range(len(s)):
            cnt = self.part(s, i, i + 1, cnt)
            cnt = self.part(s, i, i, cnt)
        return cnt

    def part(self, s, left, right, cnt):
        while (left >= 0 and right < len(s) and s[left] == s[right]):
            left -= 1
            right += 1
            cnt += 1
        return cnt
```

## 581. Shortest Unsorted Continuous Subarray

### 题目描述

```
Given an integer array, you need to find one continuous subarray that if you only sort this subarray in ascending order, then the whole array will be sorted in ascending order, too.

You need to find the shortest such subarray and output its length.

Example 1:

Input: [2, 6, 4, 8, 10, 9, 15]
Output: 5
Explanation: You need to sort [6, 4, 8, 10, 9] in ascending order to make the whole array sorted in ascending order.
```

### 解题思路

sort数列，然后遍历看第一个和最后一个和原数组不等的位置，即知道位置

### tag



### code

```python
class Solution:
    def findUnsortedSubarray(self, nums):
        res = [i for (i, (a, b)) in enumerate(zip(nums, sorted(nums))) if a != b]
        return 0 if not res else res[-1] - res[0] + 1        
```

## 560. 和为K的子数组

### 题目描述

```
给定一个整数数组和一个整数 k，你需要找到该数组中和为 k 的连续的子数组的个数。

示例 1 :

输入:nums = [1,1,1], k = 2
输出: 2 , [1,1] 与 [1,1] 为两种不同的情况。

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/subarray-sum-equals-k
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
```

### 解题思路

1.记录每个位置的到投的cnt，然后cnt[i]-cnt[j]即为其中间的和

2.从头到尾遍历，将每个位置的sum，记在map里，计数

### tag



### code

```python
#1
class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        cnt = [0]*(len(nums)+1)
        res = 0
        for i in range(1,len(nums)+1):
            cnt[i] = cnt[i-1]+nums[i-1]
        for i in range(len(nums)):
            for j in range(i+1,len(nums)+1):
                if cnt[j]-cnt[i] == k:
                    res +=1
        return res
#2
class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        dic = {0:1}
        res= 0
        s = 0
        for i in range(len(nums)):
            s += nums[i]
            if (s-k) in dic:
                res += dic[s-k]
            if s in dic:
                dic[s]+=1
            else:
                dic[s] = 1
        return res
```

## 494. Target Sum

### 题目描述

```
You are given a list of non-negative integers, a1, a2, ..., an, and a target, S. Now you have 2 symbols + and -. For each integer, you should choose one from + and - as its new symbol.

Find out how many ways to assign symbols to make sum of integers equal to target S.

Input: nums is [1, 1, 1, 1, 1], S is 3. 
Output: 5
Explanation: 

-1+1+1+1+1 = 3
+1-1+1+1+1 = 3
+1+1-1+1+1 = 3
+1+1+1-1+1 = 3
+1+1+1+1-1 = 3

There are 5 ways to assign symbols to make the sum of nums be target 3.
```

### 解题思路

1.dfs,但是会超时

2.设置每一层的记忆，如果记忆已经算过，就pass

### tag

DFS,DP,memory

### code

```python
class Solution:
    def findTargetSumWays(self, nums, S):
        index = len(nums) - 1
        curr_sum = 0
        self.memo = {}
        return self.dp(nums, S, index, curr_sum)
        
    def dp(self, nums, target, index, curr_sum):
        if (index, curr_sum) in self.memo:
            return self.memo[(index, curr_sum)]
        
        if index < 0 and curr_sum == target:
            return 1
        if index < 0:
            return 0 
        
        positive = self.dp(nums, target, index-1, curr_sum + nums[index])
        negative = self.dp(nums, target, index-1, curr_sum + -nums[index])
        
        self.memo[(index, curr_sum)] = positive + negative
        return self.memo[(index, curr_sum)]
    
#超时version
class Solution:
    def findTargetSumWays(self, nums: List[int], S: int) -> int:
        most = 0
        for num in nums:
            most += num
        dic = {nums[0]: 1, -nums[0]: 1}
        return self.part(nums[:-1], S - nums[-1]) + self.part(nums[:-1], S + nums[-1])

    def part(self, nums, S):
        if not nums:
            if S == 0:
                return 1
            else:
                return 0
        else:
            return self.part(nums[:-1], S - nums[-1]) + self.part(nums[:-1], S + nums[-1])

```



## 448. Find All Numbers Disappeared in an Array

### 题目描述

```
给定一个范围在  1 ≤ a[i] ≤ n ( n = 数组大小 ) 的 整型数组，数组中的元素一些出现了两次，另一些只出现一次。

找到所有在 [1, n] 范围之间没有出现在数组中的数字。

您能在不使用额外空间且时间复杂度为O(n)的情况下完成这个任务吗? 你可以假定返回的数组不算在额外空间内。

示例:

输入:
[4,3,2,7,8,2,3,1]

输出:
[5,6]
```

### 解题思路

1.dfs,但是会超时

2.设置每一层的记忆，如果记忆已经算过，就pass

### tag

重要点：全正数，意味着可以用正负号代表label0和1

### code

```python
class Solution:
    def findDisappearedNumbers(self, nums: List[int]) -> List[int]:
        for num in nums:
            index = abs(num)-1
            nums[index] = -abs(nums[index])
        return [i+1 for i in range(len(nums)) if nums[i]>0]
```

## 438. Find All Anagrams in a String

### 题目描述

```
Input:
s: "cbaebabacd" p: "abc"

Output:
[0, 6]

Explanation:
The substring with start index = 0 is "cba", which is an anagram of "abc".
The substring with start index = 6 is "bac", which is an anagram of "abc".
```

### 解题思路

滑动窗口，用Counter计数，跟map一样，注意当k的v==0时候，要删除键，否则会留在Counter中，影响和p的Counter的相等判断

### tag

滑动窗口

### code

```python
from collections import Counter
class Solution:
    def findAnagrams(self, s: str, p: str) -> List[int]:
        l = len(p)
        cs = Counter(s[:l-1])
        cp = Counter(p)
        i=0
        res= []
        while(i<=len(s)-l):
            cs[s[l+i-1]]+=1
            if cs==cp:
                res.append(i)
            cs[s[i]]-=1
            if cs[s[i]]==0:
                del cs[s[i]]
            i+=1
        return res
```

## 437. Path Sum III

### 题目描述

```
root = [10,5,-3,3,2,null,11,3,-2,null,1], sum = 8

      10
     /  \
    5   -3
   / \    \
  3   2   11
 / \   \
3  -2   1

Return 3. The paths that sum to 8 are:

1.  5 -> 3
2.  5 -> 2 -> 1
3. -3 -> 11
```

### 解题思路

两层，内层解决以这个点为起点的所有，外层遍历所有节点执行内层

### tag

dfs

### code

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def pathSum(self, root: TreeNode, sum: int) -> int:
        if not root:
            return 0
        def dfs(root,sum):
            count = 0
            if not root:
                return 0
            if root.val == sum:
                count+=1
            count+=dfs(root.left,sum-root.val)
            count+=dfs(root.right,sum-root.val)
            return count
        return dfs(root,sum)+self.pathSum(root.left,sum)+self.pathSum(root.right,sum)

```

## 416. Partition Equal Subset Sum

### 题目描述

```
Given a non-empty array containing only positive integers, find if the array can be partitioned into two subsets such that the sum of elements in both subsets is equal.

Note:

Each of the array element will not exceed 100.
The array size will not exceed 200.
 

Example 1:

Input: [1, 5, 11, 5]

Output: true

Explanation: The array can be partitioned as [1, 5, 5] and [11].
```

### 解题思路

背包问题，注意边界，不用res，用暂时的set即可。

### tag

背包，DP

### code

```python
class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        target = sum(nums)/2
        if target  != int(target):
            return False
        target = int(target)
        l = len(nums)
        # res = [[False]*(target+1) for i in range(l)]
        # for i in range(l):
        #     res[i][0] = True
        if nums[0]<=target:
            # res[0][nums[0]] = True
            temp = set([0])
        else:
            temp = set([0,nums[0]])
        for i in range(1,l):
            temp1 = set([])
            for t in temp:
                # res[i][t] = True
                temp1.add(t)
                if t+nums[i]<=target:
                    # res[i][t+nums[i]] = True
                    temp1.add(t+nums[i])
            temp = temp1
        return (target in temp)
```

## 437. House Robber III

### 题目描述

```
在上次打劫完一条街道之后和一圈房屋后，小偷又发现了一个新的可行窃的地区。这个地区只有一个入口，我们称之为“根”。 除了“根”之外，每栋房子有且只有一个“父“房子与之相连。一番侦察之后，聪明的小偷意识到“这个地方的所有房屋的排列类似于一棵二叉树”。 如果两个直接相连的房子在同一天晚上被打劫，房屋将自动报警。

计算在不触动警报的情况下，小偷一晚能够盗取的最高金额。

输入: [3,2,3,null,3,null,1]

     3
    / \
   2   3
    \   \ 
     3   1

输出: 7 
解释: 小偷一晚能够盗取的最高金额 = 3 + 3 + 1 = 7.

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/house-robber-iii
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
```

### 解题思路

每个节点，返回两个结果，一个是包括他的最大，一个是两个儿子的最大和

### tag

DP

### code

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def rob(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        def robInternal(root):
            if not root: return [0] * 2;
            left, right = robInternal(root.left), robInternal(root.right)
            return [max(left) + max(right), left[0] + right[0] + root.val]

        return max(robInternal(root))

```

## 309. Best Time to Buy and Sell Stock with Cooldown

### 题目描述

```
Say you have an array for which the ith element is the price of a given stock on day i.

Design an algorithm to find the maximum profit. You may complete as many transactions as you like (ie, buy one and sell one share of the stock multiple times) with the following restrictions:

You may not engage in multiple transactions at the same time (ie, you must sell the stock before you buy again).
After you sell your stock, you cannot buy stock on next day. (ie, cooldown 1 day)
Example:

Input: [1,2,3,0,2]
Output: 3 
Explanation: transactions = [buy, sell, cooldown, buy, sell]
```

### 解题思路

三个状态，hold，持有，notHold_cool：在当前节点卖了，所以告诉下一个节点cool，not_hold,不持有，下一个节点可以买或者卖

### tag

DP

### code

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        notHold,notHold_cool,hold = 0,float("-inf"),float("-inf")
        for num in prices:
            notHold,notHold_cool,hold = max(notHold,notHold_cool),hold+num,max(notHold-num,hold)
        return max(notHold, notHold_cool)
```



## 300. Longest Increasing Subsequence

### 题目描述

```
Given an unsorted array of integers, find the length of longest increasing subsequence.

Example:

Input: [10,9,2,5,3,7,101,18]
Output: 4 
Explanation: The longest increasing subsequence is [2,3,7,101], therefore the length is 4. 
```

### 解题思路

1.DP

2.DP + 二分  tail[i]为长度为i的连续子序列，最小的结束，其为上升的，所以二分

### tag

DP

### code

```python
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        if not nums:
            return 0
        res = [1] * len(nums)
        for i in range(1,len(nums)):
            m = 1
            for j in range(i-1,-1,-1):
                if nums[j]<nums[i]:
                    m = max(m,1+res[j])
            res[i] = m
        return max(res)
    
def lengthOfLIS(nums):
    tails = [0] * len(nums)
    size = 0
    for x in nums:
        i, j = 0, size
        while i != j:
            m = int((i + j) / 2)
            if tails[m] < x:
                i = m + 1
            else:
                j = m
        tails[i] = x
        size = max(i + 1, size)
    return size
```

## 287. Find the Duplicate Number

### 题目描述

```
Given an array nums containing n + 1 integers where each integer is between 1 and n (inclusive), prove that at least one duplicate number must exist. Assume that there is only one duplicate number, find the duplicate one.

Example 1:

Input: [1,3,4,2,2]
Output: 2
```

### 解题思路

```
如果数组中没有重复的数，以数组[1,3,4,2]为例，我们将数组下标n和数nums[n]建立一个映射关系f(n)，
其映射关系n->f(n)为：
0->1
1->3
2->4
3->2
我们从下标为0出发，根据f(n)计算出一个值，以这个值为新的下标，再用这个函数计算，以此类推，直到下标超界。这样可以产生一个类似链表一样的序列。
0->1->3->2->4->null

如果数组中有重复的数，以数组[1,3,4,2,2]为例,我们将数组下标n和数nums[n]建立一个映射关系f(n)，
其映射关系n->f(n)为：
0->1
1->3
2->4
3->2
4->2
同样的，我们从下标为0出发，根据f(n)计算出一个值，以这个值为新的下标，再用这个函数计算，以此类推产生一个类似链表一样的序列。
0->1->3->2->4->2->4->2->……

作者：kirsche
链接：https://leetcode-cn.com/problems/find-the-duplicate-number/solution/287xun-zhao-zhong-fu-shu-by-kirsche/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
```



### tag

环，inportant

### code

```python
class Solution:
    def findDuplicate(self, nums):
        tortoise = nums[0]
        hare = nums[0]
        while True:
            tortoise = nums[tortoise]
            hare = nums[nums[hare]]
            if tortoise == hare:
                break
        
        ptr1 = nums[0]
        ptr2 = tortoise
        while ptr1 != ptr2:
            ptr1 = nums[ptr1]
            ptr2 = nums[ptr2]
        
        return ptr1

```





# 基础算法部分

## 1.拓扑排序

### 算法思路

用字典统计每个点的入度，然后G是key是起点，value是终点的集合，zeros记录入度为0的数组。然后开始一个个pop出zeros中的点，弹出后，对应的这个点为起点的终点，入度减一，如果为0，加入zeros数组。

``` python
def topsort(G):
    in_degrees = dict((u, 0) for u in G)
    for u in G:
        for v in  G[u]:
            in_degrees[v] += 1
                                    # 每一个节点的入度
    Q = [u for u in G if in_degrees[u] == 0]
                                    # 入度为 0 的节点
    S = []
    while Q:
        u = Q.pop()
                                    # 默认从最后一个移除
        S.append(u)
        for v in G[u]:
            in_degrees[v] -= 1
                                    # 并移除其指向
            if in_degrees[v] == 0:
                Q.append(v)
    return S

G = {
    'a':'bf',
    'b':'cdf',
    'c':'d',
    'd':'ef',
    'e':'f',
    'f':''
}
# ['a', 'b', 'c', 'd', 'e', 'f']

def tp(G):
    indgrees = dict((u,0) for u in G)
    for u in G:
        for v in G[u]:
            indgrees[v]+=1
    zero = [n for n in indgrees if indgrees[n]==0]
    res = []
    while(zero):
        p = zero.pop()
        res.append(p)
        for v in G[p]:
            indgrees[v]-=1
            if  indgrees[v]==0:
                zero.append(v)
    return res

print(tp(G))

```

## 2.堆

### 算法思路：

#### 1：构造堆数列结构：

主要的部分，max_heapify函数，从最后一个非叶子节点开始遍历，保证他下面包括他的堆结构稳定。

```python
class Solution(object):
    def left(self,i):
        return 2*i+1
    def right(self,i):
        return 2*i+2
    def parent(self,i):
        return (i-2)>>1
    def max_heapify(self,nums,i):
        l,r = self.left(i),self.right(i)
        if l<len(nums) and nums[l]>nums[i]:
            largest = l
        else:
            largest = i
        if r<len(nums) and nums[r]>nums[largest]:
            largest = r
        if largest!=i:
            self.swap(nums,i,largest)
            self.max_heapify(nums,largest)
    def build_max_heap(self,nums):
        for i in range((len(nums)-2)>>1,-1,-1):
            self.max_heapify(nums,i)
    def swap(self,nums,i,j):
        temp = nums[i]
        nums[i] = nums[j]
        nums[j] = temp
    def pop_max(self,nums):
        num = nums[0]
        nums[0] = nums[-1]
        nums = nums[:len(nums)-1]
        self.max_heapify(nums, 0)
        print(nums)
        return(num,nums)

sol = Solution()
lis = [3,5,7,2]
sol.build_max_heap(lis)
m,nums = sol.pop_max(lis)
print(nums)
# print(4>>1)
# print(lis[:len(lis)-1])
```

#### 2：小顶堆，topk：

维持最大的k个，先k个建堆，然后判断最小的是否和新elem更大，否则replace他。

```python
import sys
import heapq

class TopHeap(object):
    def __init__(self,k):
        self.k = k
        self.data = []

    def push(self,elem):
        if len(self.data)<self.k:
            heapq.heappush(self.data,elem)
        else:
            mini = self.data[0]
            if elem>mini:
                heapq.heapreplace(self.data,elem)

    def topK(self):
        return [x for x in reversed([heapq.heappop(self.data) for x in range(len(self.data))])]


list_num = [1, 2, 3, 4, 5, 6, 7, 8, 9]
th = TopHeap(5)

for i in list_num:
    th.push(i)

# print (th.topK())
heapq.heapify(list_num)
print(heapq.heappop(list_num))
print(heapq.heappop(list_num))

```

## 3.sorted & sort

#### 基本概念

使用sort()方法对list排序会修改list本身,不会返回新list，通常此方法不如sorted()方便，但是如果你不需要保留原来的list，此方法将更有效sort()。

sort()不能对dict字典进行排序。

sorted对原数组不做改变

```python
my_list = [3, 5, 1, 4, 2]
result = sorted(my_list)
result2 = sorted(my_list,reverse=True)
print (result)
print(my_list)

#[1, 2, 3, 4, 5]
#[5, 4, 3, 2, 1]
#[3, 5, 1, 4, 2]
```

#### key

``` python
student_tuples = [
        ('john', 'A', 15),
        ('jane', 'B', 12),
        ('dave', 'B', 10),
]

print(sorted(student_tuples,key = lambda student:student[2],reverse = True))
# [('john', 'A', 15), ('jane', 'B', 12), ('dave', 'B', 10)]
```

dict part

```python
my_dict = {"a":"2", "c":"5", "b":"1"}

result = sorted(my_dict)
print(result)
result2 = sorted(my_dict,key= lambda key:my_dict[key])
print(result2)
```

## 4.lambda表达式

```python
g = lambda x:x+1
print(g(1))
# 2

print(list(map(lambda x,y:x+y,[1,3],[2,4])))
# [3, 7]

li = [11, 22, 33]
new_list = filter(lambda arg: arg > 22, li)
print(list(new_list))
#[33]



```

## 5.树

### 中序

左边一直入栈，不存在后pop，root变为右边。

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        res = []
        stack = []
        while(True):
            while(root):
                stack.append(root)
                root = root.left
            if not stack:
                return res
            node = stack.pop()
            res.append(node.val)
            root = node.right       
```

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def preorderTraversal(self, root: TreeNode) -> List[int]:
        stack = [root]
        res = []
        while stack:
            node = stack.pop()
            if node:
                res.append(node.val)
                stack.append(node.right)
                stack.append(node.left)
        return res
```

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
###  第二次访问进res，还是先进右边，再进左边。
class Solution:
    def postorderTraversal(self, root: TreeNode) -> List[int]:
        res = []
        stack = [(root,False)]
        while(stack):
            node,flag = stack.pop()
            if node:
                if flag:
                    res.append(node.val)
                else:
                    stack.append((node,True))
                    stack.append((node.right,False))
                    stack.append((node.left,False))
        return res
```

