# kaggle

## 注册

###  绑定时的小坑

![img](https://pic1.zhimg.com/80/v2-02b5b7ba668cae7f8240f74a24d275bc_hd.jpg)

第一开始，也不知是没翻qiang的缘故，还是浏览器的缘故，人机交互的按钮一直刷不出来，故一直无法成功发送验证码，而后开了代理就成功了，这里手机号的话+86后跟空格再加国内的手机号应该即可。群内的大佬们说有的验证码很久才能收到，所以推荐大家用google账号关联，这种方法应该比较稳定。

### 参加比赛

1. fork and edit

随后的话，点进我们的比赛链接，点击join competition就可加入比赛。由于我一直没有做过什么大型的完整的nlp项目，所以暂时可能还需要一定阶段来适应，所以大佬建议可以先用其他人写好的kernel去跑一个基础分，适应一下流程，而且也可以随后进行edit，所以我们点击kernel，我选择了这一个进行测试，

![img](https://pic3.zhimg.com/80/v2-9f43c03c1caf9dd443ef94bb4cd2fc9e_hd.png)

进入kernel后，点击Copy and Edit将该kernel归入我们的个人空间中，随后点右上角的commit，待执行完毕。

最后进入kernels-Your Work，选中你要提交的work，进入该work的output，submit to competition即可，最后在最外面的My submissions可以看到我们项目的得分。

2. create a new kernel

   有两种，jupyter notebook 或者 script，用python就可以

### 主要汲取知识的地方

1. kernel区

   kernel区有各种优秀的人分享的baseline，可以基于这些baseline去思考，优化，做ensemble。

2. discussion区

   可以在该区求助，或者看别人的讨论，有很多大神的idea会在其中分享，或者给一些模糊的方向，给参赛者们去追逐。

3. 由于要有一个比较公平的竞赛环境，kernel赛基本都是要求在kaggle自己的环境下运行，所以一样的内存，一样的gpu（有的kernel赛比如jigsaw可以将数据在场外训练后模型重新导入），一样的运行时间限制，比如jigsaw这个比赛的话要求使用gpu的话2小时，cpu的话9小时，这样的话就可以锻炼对内存的控制以及gpu的属性了解

   