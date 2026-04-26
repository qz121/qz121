# 模型感知的神经元遗忘学习  
论文发表：ACL 2025  
数据集：MLLMU-Bench、LLaVA-Bench、MMMA  
github链接：https://github.com/franciscoliu/MANU  


## 一、论文背景

已有很多机器遗忘方法主要面向 **纯文本 LLM**，直接迁移到 **多模态模型 MLLM** 时会遇到新问题。  
因为在 MLLM 中，知识不是只存在于文本通道里，而是分布并纠缠在 **文本模态** 和 **视觉模态** 中，导致遗忘变得更困难。

---

## 二、论文要解决的核心问题

这篇论文关注的核心问题是：

> **如何让多模态大语言模型在“文本输入”和“图文输入”两种情况下，都真正遗忘目标知识，而不是只在其中一种模态上遗忘成功？**

论文指出，现有方法常出现一种现象：

- 对图文输入，模型好像已经“忘了”
- 但对纯文本输入，模型仍然保留相关知识

这说明 MLLM 中存在 **跨模态知识纠缠（entangled knowledge）** 问题。不同模态输入会激活不同神经元，因此如果只针对一种输入形式做遗忘，另一种形式下对应知识可能依然存在。

### 论文具体聚焦的三个问题

1. **目标知识能否被有效遗忘？**
2. **能否解决多模态下遗忘不平衡的问题？**
3. **能否在遗忘目标知识的同时，尽量保留模型原有能力（model utility）？**

---

## 三、论文的核心思想

论文提出了一种新的方法：

# MANU
**Modality-Aware Neuron Unlearning**

其核心思想是：

> 不再单纯依赖参数更新去“逼模型忘记”，而是从神经元层面出发，找出那些与待遗忘知识最相关、且在不同模态中起关键作用的神经元，然后进行有选择的剪枝（pruning）。主要剪MLP层(Multi-Layer Perceptron layer)，中文常叫多层感知机层。在 Transformer / LLM / MLLM 里，它通常也被叫作 FFN 层，也就是 Feed-Forward Network，前馈网络层。负责信息加工（拿到这些上下文信息后，我应该如何理解、变换、激活相关知识？）。

换句话说，MANU 的重点是：

- 分析哪些神经元和 forget set 最相关
- 分析这些神经元在文本 / 多模态中的作用差异
- 保留对 retain set 有帮助的神经元
- 剪掉对遗忘目标最关键的那部分神经元

---

## 四、技术路线

MANU 的整体流程可以概括为两个阶段：

1. **重要神经元选择（Important Neuron Selection）**
2. **选择性剪枝（Selective Pruning）**

---

## 五、阶段一：重要神经元选择

### 1. 数据划分

论文把数据分为两类：

- **Forget Set**：希望模型遗忘的数据，记作 $D_f$
- **Retain Set**：希望模型继续保留的数据，记作 $D_r$

同时，又把输入形式拆成两种：

- **文本输入（text-only）**
- **多模态输入（image + text）**

记作：

- $D_{\text{text}} \subset D$
- $D_{\text{multi}} \subset D$

这样做的目的是比较同一批知识在不同模态下对应的神经元激活差异。

---

### 2. 分析对象

论文主要分析：

- 语言模块中的 MLP 神经元
- 视觉模块中的 MLP 神经元

作者认为，MLP 层是模型内部存储和表达知识的重要位置，因此适合作为遗忘操作的主要目标。

---

### 3. 四种重要性函数

为了判断一个神经元是否和待遗忘知识强相关，论文设计了四类重要性指标。

#### （1）绝对重要性 $I_{\text{abs}}$

衡量同一个神经元在不同模态下的**平均激活强度差异**。

$$
I_{\text{abs}}(D, n) := 
\frac{|\bar{Z}_{\text{multi}} - \bar{Z}_{\text{text}}|}
{\bar{Z}_{\text{multi}} + \bar{Z}_{\text{text}} + \epsilon}
$$

其中：

$$
\bar{Z}_{\text{multi}} =
\frac{1}{|D_{\text{multi}}|}
\sum_{d \in D_{\text{multi}}} |z_{\text{multi}}(d)|
$$

$$
\bar{Z}_{\text{text}} =
\frac{1}{|D_{\text{text}}|}
\sum_{d \in D_{\text{text}}} |z_{\text{text}}(d)|
$$

含义是：

- 若某神经元在图文输入和文本输入中的平均激活差异很大
- 则它可能承担模态特定的信息处理作用

---

#### （2）频率重要性 $I_{\text{freq}}$

  衡量神经元在不同模态下**显著激活的频率差异**。  
**但是论文中并没有体现阈值是多少也没有交代阈值是怎么取的？但是代码中阈值为0.1**

先定义模态特定的激活频率：

$$
N_{\text{multi}} =
\left|
\{ d \in D_{\text{multi}} \mid |z_{\text{multi}}(d)| > \tau \}
\right|
$$

$$
N_{\text{text}} =
\left|
\{ d \in D_{\text{text}} \mid |z_{\text{text}}(d)| > \tau \}
\right|
$$

再定义：

$$
I_{\text{freq}}(D, n) :=
\frac{|\Delta N|}{\Sigma N + \epsilon}
$$

其中：

$$
\Delta N = N_{\text{multi}} - N_{\text{text}}, \qquad
\Sigma N = N_{\text{multi}} + N_{\text{text}}
$$

直觉是：

- 有些神经元不是激活值特别大
- 但它会在某一模态中经常被触发
- 这种“稳定激活”也说明它很重要

---

#### （3）方差重要性 $I_{\text{var}}$

衡量神经元在不同模态中的**激活分布变化程度**。

先定义方差：

$$
\mathrm{Var}_{\text{multi}} =
\frac{1}{|D_{\text{multi}}|}
\sum_{d \in D_{\text{multi}}}
\left(z_{\text{multi}}(d) - \bar{Z}_{\text{multi}}\right)^2
$$

$$
\mathrm{Var}_{\text{text}} =
\frac{1}{|D_{\text{text}}|}
\sum_{d \in D_{\text{text}}}
\left(z_{\text{text}}(d) - \bar{Z}_{\text{text}}\right)^2
$$

然后：

$$
I_{\text{var}}(D, n) := \sqrt{\mathrm{Var}_{\text{multi}} + \mathrm{Var}_{\text{text}}}
$$

直觉是：

- 一个神经元若在不同模态中具有更丰富、更分散的响应
- 就更可能承载更有信息量的模式
- 而不是无意义地长期接近 0

---

#### （4）均方根重要性 $I_{\text{rms}}$

衡量神经元是否具有**持续较强、且具有模态区分度的激活**。

$$
I_{\text{rms}}(D, n) :=
\sqrt{
\frac{|\Delta Z^2|}{\Sigma Z^2 + \epsilon}
}
$$

其中：

$$
Z_{\text{multi}}^2 = \sum_{d \in D_{\text{multi}}} z_{\text{multi}}(d)^2
$$

$$
Z_{\text{text}}^2 = \sum_{d \in D_{\text{text}}} z_{\text{text}}(d)^2
$$

$$
\Delta Z^2 = Z_{\text{multi}}^2 - Z_{\text{text}}^2, \qquad
\Sigma Z^2 = Z_{\text{multi}}^2 + Z_{\text{text}}^2
$$

直觉是：

- 有些神经元经常激活，但可能只是冗余激活
- 这个指标想保留真正“持续强激活且有区分性”的神经元
- 同时抑制冗余神经元

---

### 4. 综合重要性

论文把四个重要性函数聚合起来，形成统一的重要性度量：

$$
I(D, n) := \sum_{k \in K} I_k(D, n)
$$

其中：

$$
K = \{ I_{\text{abs}}, I_{\text{freq}}, I_{\text{var}}, I_{\text{rms}} \}
$$

这个统一分数综合考虑了：

- 激活强度差异
- 激活频率差异
- 激活分布差异
- 持续强激活与冗余程度

---

## 六、阶段二：选择性剪枝

在第一阶段得到每个神经元的重要性后，论文进一步定义了一个打分函数，用来比较：

- 该神经元对 **Forget Set** 的重要程度
- 该神经元对 **Retain Set** 的重要程度

### 1. 神经元打分函数

$$
S_n =
\frac{I(D_f, n)}{I(D_r, n) + \epsilon}
$$

含义是：

- 若一个神经元对 forget data 很重要
- 但对 retain data 不那么重要
- 那么它的分数就会更高，更适合被剪掉

---

### 2. 选择要剪枝的神经元集合

给定一个剪枝率 $\alpha$，选择分数最高的前 $\alpha\%$ 神经元


---

### 3. 剪枝操作

设原始模型为 $\theta$，剪枝后模型为 $\theta'$，则论文写作：

$$
\theta' =
\begin{cases}
0, & \text{if } n \in \mathcal{N} \\
\theta, & \text{otherwise}
\end{cases}
$$

在直观上，这表示：

- 对被选中的神经元，相关权重置零
- 其余参数保持不变

---

## 七、论文的关键创新点

### 创新点 1：提出了面向 MLLM 的模态感知遗忘框架
已有很多方法面向 LLM，但没有专门解决 MLLM 中“文本-视觉知识纠缠”的问题。MANU 的贡献在于显式考虑了模态差异。

### 创新点 2：从“神经元级别”做遗忘
不是简单通过训练目标去逼迫模型忘记，而是直接定位和目标知识相关的神经元，再做剪枝。

### 创新点 3：兼顾遗忘效果与模型效用
MANU 不只是追求“忘得更多”，也强调尽量保住：
- retain set 表现
- 真实名人集合表现
- 通用推理与帮助能力

---

## 八、基线方法中的主要公式

除了 MANU，论文附录里还给出了几个基线方法的目标函数。

### 1. Gradient Ascent（GA）

其目标是在 forget set 上**增大损失**：

$$
L(D_f, w) =
\frac{1}{|D_f|}
\sum_{x \in D_f} \ell(x, w)
$$

这里：

- $x \in D_f$ 表示 forget set 中的样本
- $\ell(x, w)$ 表示模型参数为 $w$ 时对样本 $x$ 的损失

---

### 2. Gradient Difference

在遗忘 forget set 的同时，尽量保留 retain set 的性能：

$$
L_{\text{diff}} = -L(D_f, w) + L(D_r, w)
$$

含义是：

- 第一项推动模型忘记 $D_f$
- 第二项推动模型保留 $D_r$

---

### 3. KL Minimization

让当前模型在 retain set 上尽量接近原模型，同时偏离 forget set：

$$
L_{\text{KL}} =
-L(D_f, w)
+
\frac{1}{|D_r|}
\sum_{s \in D_r}
\mathrm{KL}(M_o \Vert M_c)(s)
$$

其中：

- $M_o$ 是原始模型
- $M_c$ 是当前模型
- $\mathrm{KL}(M_o \Vert M_c)(s)$ 表示在样本 $s$ 上两者输出分布的 KL 散度

---

### 4. Negative Preference Optimization（NPO）

论文给出的 NPO 目标为：

$$
L_{\text{NPO}} =
\frac{2}{\beta}
\mathbb{E}_{(x,y)\in D_f}
\left[
\log
\left(
1 +
\left(
\frac{\pi_\theta(y \mid x)}
{\pi_{\text{ref}}(y \mid x)}
\right)^\beta
\right)
\right]
$$

其中：

- $\pi_\theta(y \mid x)$ 是当前模型对 token $y$ 的预测概率
- $\pi_{\text{ref}}(y \mid x)$ 是参考模型的预测概率
- $\beta$ 是平滑参数

这个目标的思路是：

- 把 forget set 当作“不希望偏好”的数据
- 通过偏好优化方式降低模型对这些数据的依赖

---

## 九、实验想回答什么问题

论文实验围绕以下几个问题展开：

### 1. MANU 能否有效遗忘目标知识？
作者在 Forget Set 和 Test Set 上评估分类、生成、完形填空等任务，检查模型是否真的不再记得目标资料。

### 2. MANU 能否缓解跨模态遗忘不平衡？
作者分别测试：
- 图文输入
- 纯文本输入

观察遗忘是否在两种输入形式下都成立。

### 3. 不同剪枝比例有什么影响？
作者比较 2%、5%、10% 剪枝比例，分析遗忘强度与模型效用之间的权衡。

### 4. MANU 能否平衡“遗忘”和“保留能力”？
除了 Forget/Test Set，还用：
- Retain Set
- Real Celebrity Set
- MMMU
- LLaVA-Bench

来评估模型通用能力是否被破坏。

---

## 十、论文结论

论文的主要结论可以概括为：

1. **MANU 能更有效地实现多模态遗忘**
   - 尤其比一些传统方法更能同时覆盖文本和图文两种输入形式

2. **MANU 能更好地缓解遗忘不平衡问题**
   - 传统方法常常“图文忘了，文本没忘”
   - MANU 在两种模态上更均衡

3. **MANU 在遗忘与模型效用之间取得了更好的折中**
   - 虽然某些 GA 类方法可能忘得更狠
   - 但通常会严重损伤 retain 和 general utility
   - MANU 在整体上更平衡

4. **剪枝比例越高，遗忘越强，但副作用也越大**
   - 剪得越多，forget/test 表现下降越明显
   - 但 retain/real celebrity/general benchmarks 也更容易受损

---

## 十一、论文存在的问题与局限

论文也明确承认了一些限制：

### 1. 应用范围还比较有限
当前实验主要聚焦于 MLLMU-Bench 中的虚构人物资料遗忘，尚未充分验证：
- 有害回答遗忘
- 版权内容遗忘
- 其他真实应用场景

### 2. 鲁棒性还需要进一步验证
虽然论文用了多种评估指标，但对更强攻击或更复杂恢复方式下的鲁棒性仍需研究。

### 3. 剪枝比例较敏感
剪枝比例会显著影响：
- 遗忘效果
- 模型稳定性
- utility 保留程度

这说明方法还没有完全解决“最优平衡点”问题。

### 4. 仍然存在额外计算成本
虽然它不需要从头重训，也不依赖大量梯度更新，但在筛选神经元时，仍需要对不同数据集、不同模态下的神经元激活做系统统计，因此离线成本并不低。


---
## 十二、下一步工作
### 1、结构化神经元剪枝？

---
