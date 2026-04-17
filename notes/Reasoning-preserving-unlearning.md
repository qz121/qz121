# Towards Reasoning-Preserving Unlearning in Multimodal Large Language Models
论文链接：https://arxiv.org/pdf/2512.17911
---

## 1. 背景

这篇论文讨论的是：**带显式推理链（chain-of-thought）的多模态大模型，应该怎么做 machine unlearning**。

普通 unlearning 往往主要看一件事：

- 模型还能不能在 forget set 上答对原来的答案

但这篇论文指出，对 **Reasoning Multimodal Large Language Models（RMLLMs）** 来说，这样的标准不够。因为即使模型把最终答案改掉了，**中间推理过程**里仍然可能把被要求遗忘的信息泄露出来。另一方面，如果你干预得太重，又会把模型原本的 reasoning 能力一起破坏掉。

论文把这两个问题分别称为：

### 1.1 Reasoning Leakage
答案变了，但 reasoning 仍然泄露敏感信息。

### 1.2 Reasoning Retention
为了压制泄露而进行的干预，伤害了正常推理能力。

所以这篇论文真正追求的目标不是“只忘掉答案”，而是：

- **忘掉答案**
- **忘掉 reasoning trace**
- **尽量保住一般推理能力**

---

## 2. 论文的两大贡献

### 2.1 提出新的 benchmark：RMLLMU-Bench

作者认为，以前的 MLLM forgetting benchmark 主要测 final answer 有没有变，不直接衡量 reasoning 过程有没有泄露，也不衡量 reasoning 能力保留得怎么样。于是他们提出了 **RMLLMU-Bench**，把 reasoning 过程也纳入评测。

### 2.2 提出新的方法：R-MUSE

作者的方法叫 **R-MUSE**，全称是 *Reasoning-preserving MLLM Unlearning via Subspace guidance and Adaptive StEering*。

它的特点是：

- **training-free**
- **inference-time**
- 不靠重新训练模型参数
- 而是在推理时对 hidden representation 做 steering

---

## 3. RMLLMU-Bench 到底多了什么

RMLLMU-Bench 是在原来的 MLLMU-Bench 上扩展出来的。作者给每个样本增加了一段结构化 reasoning，并要求这段 reasoning 满足三条原则：

### 3.1 Attributability（可归因）
每一步 reasoning 都必须能追溯到证据，比如：

- profile 字段
- 图像区域

### 3.2 Conservativeness（保守性）
不能乱用外部世界知识，只能根据给定输入推理。

### 3.3 Consistency（一致性）
推理链本身要逻辑连贯，而且要真正支持最终答案。

---

## 4. 论文新加的两个关键指标

### 4.1 RIL：Reasoning Information Leakage

RIL 衡量的是：**模型在 reasoning 里还泄不泄露被忘记的信息**。

作者设计了两层检测：

#### 显式泄露
例如忘记信息是 `residence: Japan`，reasoning 里直接出现 `Japan`。

#### 隐式泄露
例如 reasoning 里写 `He lives in Tokyo`，虽然没直接写 Japan，但依然算泄露。

最终 RIL 把这两部分加权合成。

所以：

- **RIL 越低越好**

### 4.2 RCR：Reasoning Capability Retention

RCR 衡量的是：**在非 forget 数据上，模型的 reasoning 还保留得怎么样**。

也就是看：

- 逻辑是否还通顺
- 证据是否还站得住
- 是否还能正常推理

所以：

- **RCR 越高越好**

---

## 5. R-MUSE 的整体思路

作者把方法概括成三个问题：

1. **What to steer**：要改哪种方向
2. **Where to steer**：哪些方向不能乱动
3. **How strong to steer**：每次该改多大

对应地，R-MUSE 也分成三块：

### 5.1 Span Hybrid Unlearning Subspace
用来找“遗忘方向”。

### 5.2 Reasoning Retain Subspace（RRS）
用来保护正常 reasoning。

### 5.3 Adaptive Calibration Steering（ACS）
用来决定这次 steering 的强度。

---

## 6. 第一块：Span Hybrid Unlearning Subspace

作者不是只看 final answer token，而是同时看两部分：

- **Answer span**
- **Reasoning / CoT span**

### 为什么要同时看这两部分？

因为如果只盯答案，模型可能只是把最后一句话改了，但 reasoning 里仍然保留了记忆痕迹。

作者的做法是：

- 为 forget 样本构造 refusal-style 的正样本
- 把模型原本的 answer-form / reasoning-form 输出当负样本
- 分别对 **Answer span** 和 **CoT span** 做差分表示
- 做标准化后合并
- 再做 SVD，提取主要子空间

这个子空间就是 **unlearning subspace**。

你可以把它理解成：

> “模型里和被遗忘知识及其 reasoning trace 最相关的方向集合”

---

## 7. 第二块：RRS（Reasoning Retain Subspace）

这部分是整篇论文最关键的保护装置。

作者发现，如果你对所有输入都沿着 unlearning subspace 去改，很容易把模型正常 reasoning 的能力一起伤掉。

所以他们又在 retain set 上学了一个子空间：**Reasoning Retain Subspace（RRS）**。

### 它怎么学出来？

对 retain 样本，构造一对输出：

- 一个有 step-by-step reasoning
- 一个只有 direct answer

然后做和前面类似的差分 + SVD。得到的就是“支撑正常 reasoning 的方向”。

### RRS 有两个作用

#### 作用 1：做 gate
如果当前 query 和 RRS 很对齐，说明它更像正常 reasoning query，这时就没必要激进 unlearn。

#### 作用 2：做正交保护
真正的 steering 方向，不是直接拿原始遗忘方向就用，而是要**投影到 RRS 的正交补**上。

这句话特别重要：

> **投影决定“哪些方向能动，哪些方向不能动”。**

也就是：

- RRS 里的方向尽量不碰
- 只在 RRS 的正交补里动

---

## 8. 第三块：ACS（Adaptive Calibration Steering）

这部分解决的是：

> **已经知道该往哪边动了，那到底动多大？**

传统 steering 往往会写成：

```math
\tilde h = h + \lambda f(h)
