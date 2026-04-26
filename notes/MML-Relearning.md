# 1. 论文基本信息

**论文标题**：  
Towards LLM Unlearning Resilient to Relearning Attacks: A Sharpness-Aware Minimization Perspective and Beyond

**中文理解**：  
面向抵抗重学习攻击的大语言模型遗忘：基于锐度感知最小化的视角及扩展

**GitHub 链接**：  
https://github.com/OPTML-Group/Unlearn-Smooth

**发表 venue**：  
**ICML 2025**。

**完整发表信息**：  
Proceedings of the 42nd International Conference on Machine Learning, Vancouver, Canada, PMLR 267, 2025.


---



---

## 2. 论文关注的核心问题

已有 LLM Unlearning 方法虽然可以让模型在当前状态下“看起来忘掉”某些知识，但存在一个重要漏洞：

> 模型可能会被很少量的数据重新训练后，再次恢复被遗忘的知识。

这种攻击叫做：

**Relearning Attack，重学习攻击。**

也就是说，模型虽然完成了 unlearning，但攻击者只要拿少量 forget data 对模型进行 fine-tuning，就可能让模型重新学回被遗忘的内容。

---

## 3. Fine-tuning 是什么？

**Fine-tuning** 中文叫 **微调**。

它指的是：

> 在一个已经训练好的模型基础上，使用新的特定数据继续训练，让模型朝某个任务、领域或目标方向调整。

例如：

```text
通用大模型
    ↓
使用医学问答数据 fine-tune
    ↓
更擅长医学问答的模型
```

在这篇论文中，fine-tuning 被攻击者用作一种攻击手段：

```text
模型已经通过 unlearning 忘掉某些知识
    ↓
攻击者使用少量 forget data 对模型 fine-tune
    ↓
模型可能重新学回这些被遗忘的知识
```

因此，本文研究的不是普通 fine-tuning，而是：

> 如何防止模型在 unlearning 后被少量 fine-tuning 数据重新恢复知识。

---

## 4. Relearning Attack 建模

假设模型经过 unlearning 后得到参数：

$$
\theta_u
$$

攻击者使用少量 forget data 的子集：

$$
D'_f \subset D_f
$$

对模型进行重新训练，引入权重更新：

$$
\delta
$$

攻击目标可以表示为：

$$
\min_\delta \ell_{relearn}(\theta_u + \delta | D'_f)
$$

其中：

- $\theta_u$ 是已经 unlearned 的模型参数；
- $\delta$ 是攻击者通过 fine-tuning 引入的参数变化；
- $D'_f$ 是少量用于重学习的数据；
- $\ell_{relearn}$ 是重新学习目标。

直观理解：

```text
Unlearning：
把模型从掌握某些知识的状态推开

Relearning Attack：
用少量数据把模型再拉回原来的知识状态
```

---

## 5. 论文的核心思想

论文认为，已有 unlearning 方法的问题在于：

> 它们只让模型在当前参数点上忘掉知识，但没有保证模型在参数附近区域也保持遗忘。

也就是说，模型可能处在一个“尖锐”的遗忘点：

```text
当前参数点：
模型确实忘掉了

稍微 fine-tune 一下：
模型又恢复了被遗忘知识
```

论文希望模型达到一种更“平坦”的遗忘状态：

```text
当前参数点：
模型忘掉了

参数稍微变化：
模型仍然忘掉

攻击者需要更大代价：
才能重新恢复知识
```

因此，作者提出从 **loss landscape smoothness** 的角度增强 unlearning 鲁棒性。

---

## 6. 技术核心：将 Relearning Attack 看成权重空间扰动

传统 adversarial training 通常考虑的是输入扰动，例如：

```text
输入 x 被攻击成 x + δ
```

而本文认为，relearning attack 本质上不是输入变化，而是模型权重变化：

```text
模型参数 θ 被攻击成 θ + δ
```

因此，作者将 relearning attack 建模为：

**Weight-space perturbation，权重空间扰动。**

对比：

| 方法 | 攻击对象 | 扰动空间 |
|---|---|---|
| 传统对抗训练 | 输入样本 | Input space |
| 本文方法 | 模型参数 | Weight space |

---

## 7. 鲁棒 Unlearning 的 Min-Max 优化目标

为了让模型抵抗权重扰动，作者将 unlearning 建模成一个 min-max 鲁棒优化问题：

$$
\min_\theta \max_{\|\delta\|_p \leq \rho}
\ell_f(\theta + \delta | D_f)
+
\lambda \ell_r(\theta | D_r)
$$

其中：

- 外层 $\min_\theta$：执行 unlearning；
- 内层 $\max_\delta$：模拟最坏情况下的 relearning 攻击；
- $\delta$：模型参数扰动；
- $\rho$：限制扰动大小；
- $\ell_f(\theta + \delta | D_f)$：要求模型在参数扰动后仍然保持遗忘；
- $\ell_r(\theta | D_r)$：保持模型在 retain set 上的能力。

这个目标的含义是：

> 模型不仅要在当前参数下忘掉，还要在最坏的参数扰动下仍然保持遗忘。

---

## 8. 核心技术：Sharpness-Aware Minimization, SAM

论文发现，上面的 min-max 鲁棒优化目标和 **SAM** 的思想高度一致。

SAM 的全称是：

**Sharpness-Aware Minimization**

中文可以理解为：

**锐度感知最小化**

SAM 原本用于提升模型泛化能力，其核心思想是：

> 不仅要让当前参数点的 loss 好，还要让参数附近区域的 worst-case loss 也好。

应用到 LLM Unlearning 中，SAM 的含义变成：

> 不仅要让模型当前参数点忘掉知识，还要让模型在参数附近区域都保持遗忘。

---

## 9. SAM 在本文中的具体形式

SAM-enhanced forget loss 可以写成：

$$
\ell_f^{SAM}(\theta)
=
\max_{\|\delta\|_2 \leq \rho}
\ell_f(\theta + \delta)
$$

这个公式表示：

> 在模型参数附近寻找一个最坏扰动，并要求模型在这个扰动下仍然完成遗忘。

直接求解这个最大化问题成本较高，因此论文使用一阶 Taylor 近似：

$$
\ell_f(\theta + \delta)
\approx
\ell_f(\theta)
+
\delta^T \nabla_\theta \ell_f(\theta)
$$

在约束 $\|\delta\|_2 \leq \rho$ 下，最坏扰动方向为：

$$
\delta^*(\theta)
=
\rho
\frac{\nabla_\theta \ell_f(\theta)}
{\|\nabla_\theta \ell_f(\theta)\|_2}
$$

也就是说：

> 最坏的权重扰动方向，就是 forget loss 梯度方向。

---

## 10. SAM-enhanced Unlearning 训练流程

论文中的 SAM-enhanced unlearning 可以概括为以下步骤：

### Step 1：初始化模型

从原始模型参数开始：

$$
\theta_u \leftarrow \theta
$$

---

### Step 2：采样 forget data

从 forget set 中采样一批数据：

$$
(x_f, y_f) \sim D_f
$$

计算 forget loss 梯度：

$$
\nabla_\theta \ell_f(\theta_u; x_f, y_f)
$$

---

### Step 3：构造 SAM 权重扰动

根据 forget loss 梯度构造扰动：

$$
\delta =
\rho
\frac{\nabla_\theta \ell_f(\theta_u)}
{\|\nabla_\theta \ell_f(\theta_u)\|_2}
$$

---

### Step 4：在扰动后的模型上计算 forget gradient

使用扰动后的参数：

$$
\theta_u + \delta
$$

计算新的 forget gradient：

$$
g_f =
\nabla_\theta \ell_f(\theta_u + \delta)
$$

---

### Step 5：采样 retain data

从 retain set 中采样数据：

$$
(x_r, y_r) \sim D_r
$$

计算 retain gradient：

$$
g_r =
\nabla_\theta \ell_r(\theta_u)
$$

---

### Step 6：联合更新模型

最终更新为：

$$
\theta_u
\leftarrow
\theta_u
-
\eta(g_f + \lambda g_r)
$$

其中：

- $g_f$ 用于推动模型鲁棒遗忘；
- $g_r$ 用于保持模型原有能力；
- $\eta$ 是学习率；
- $\lambda$ 是 retain loss 权重。

---

## 11. 为什么 SAM 可以抵抗 Relearning Attack？

论文的解释是：

> SAM 可以让 forget loss landscape 更平滑，从而让模型的遗忘状态更加稳定。

普通 unlearning 可能得到一个尖锐解：

```text
当前点遗忘成功
    ↓
参数稍微变化
    ↓
遗忘效果快速消失
    ↓
模型容易被 relearning attack 恢复
```

SAM 会得到一个更平坦的解：

```text
当前点遗忘成功
    ↓
参数附近区域也保持遗忘
    ↓
少量 fine-tuning 不容易恢复知识
    ↓
模型更鲁棒
```

因此，SAM 的本质作用是：

> 将“点状遗忘”变成“区域性遗忘”。

---

## 12. 从 SAM 到更广义的 Smoothness Optimization

论文不仅研究 SAM，还进一步研究了其他平滑优化方法。

作者认为：

> 如果鲁棒 unlearning 的关键是平滑 forget loss landscape，那么其他 smoothness optimization 技术也应该有效。

论文测试了以下方法：

| 方法 | 中文理解 |
|---|---|
| SAM | 锐度感知最小化 |
| RS | 随机平滑 |
| GP | 梯度惩罚 |
| CR | 曲率正则化 |
| WA | 权重平均 |

---

## 13. Randomized Smoothing, RS

RS 不寻找最坏扰动，而是对随机扰动求平均：

$$
\ell_f^{RS}(\theta)
=
\mathbb{E}_{\delta \sim \mathcal{N}(0, \sigma^2)}
[
\ell_f(\theta + \delta)
]
$$

它的目标是：

> 让模型在随机权重扰动下也保持稳定遗忘。

---

## 14. Gradient Penalty, GP

GP 直接惩罚 forget loss 的梯度范数：

$$
\ell_f^{GP}(\theta)
=
\ell_f(\theta)
+
\rho \|\nabla_\theta \ell_f(\theta)\|_2
$$

其含义是：

> 如果 forget loss 对参数变化过于敏感，就进行惩罚。

这和 SAM 的一阶近似非常相关。

---

## 15. Curvature Regularization, CR

CR 直接约束 loss landscape 的曲率变化：

$$
\ell_f^{CR}(\theta)
=
\ell_f(\theta)
+
\gamma
\|
\nabla_\theta \ell_f(\theta + \mu v)
-
\nabla_\theta \ell_f(\theta)
\|_2
$$

它通过限制梯度变化，使 loss surface 更平滑。

---

## 16. Weight Averaging, WA

WA 对训练过程中的多个模型 checkpoint 进行权重平均：

$$
\theta_{WA,t}
=
\frac{\theta_{WA,t} \cdot n + \theta_t}{n+1}
$$

其思想是：

> 权重平均可以让模型落到更宽、更平坦的参数区域，从而提高稳定性。

---

## 17. 与已有 Unlearning 方法的结合

这篇论文不是完全重新设计一个 unlearning 方法，而是将 SAM 作为一个增强模块，插入已有 unlearning 方法中。

论文测试了以下组合：

| 原始方法 | SAM 增强版本 |
|---|---|
| NPO | NPO + SAM |
| GradDiff | GradDiff + SAM |
| RMU | RMU + SAM |

其中重点方法是：

**NPO + SAM**

NPO 是已有的强 unlearning baseline，但容易受到 relearning attack。  
加入 SAM 后，NPO 不仅能让模型在当前参数下遗忘，还能让模型在参数邻域中保持遗忘。

---

## 18. 实验设置

论文主要在两个 benchmark 上进行实验：

### 18.1 WMDP

WMDP 是 Weapons of Mass Destruction Proxy benchmark，用于测试模型是否会输出危险领域知识。

论文主要关注 WMDP Bio，也就是生物安全相关危险知识。

评价指标是：

$$
UE = 1 - Accuracy
$$

其中：

- Accuracy 越低，说明模型越不会回答危险问题；
- UE 越高，说明 unlearning 效果越好。

---

### 18.2 MUSE

MUSE 用于评估模型是否遗忘版权文本内容。

包括两个任务：

| 任务 | 内容 |
|---|---|
| Books | Harry Potter 书籍文本 |
| News | BBC 新闻文本 |

评价指标包括：

- KnowMem：知识记忆程度；
- VerbMem：逐字记忆程度。

在 MUSE 中，KnowMem 和 VerbMem 越低，说明遗忘效果越好。

---

## 19. 实验结论

论文实验表明：

### 21.1 NPO + SAM 显著提升抗 Relearning Attack 能力

普通 NPO 在使用少量 forget samples 重新 fine-tune 后，遗忘效果明显下降。

NPO + SAM 在相同攻击下仍然保持更好的 unlearning effectiveness。

---

### 21.2 SAM 不明显损害模型通用能力

加入 SAM 后，模型在 retain set 或 MMLU 上的能力没有明显下降。

这说明 SAM 在提升鲁棒性的同时，基本保留了模型原有能力。

---

### 21.3 SAM 对多种 Unlearning 方法都有效

SAM 不仅增强了 NPO，也增强了 GradDiff 和 RMU。

这说明 SAM 不是只对某一个方法有效，而是一种比较通用的鲁棒性增强技术。

---

### 21.4 SAM 对 Jailbreaking Attack 也有帮助

论文还测试了输入层面的 jailbreaking attack。

结果发现：

- 普通 NPO 容易被 adversarial prompt 诱导输出被遗忘知识；
- NPO + SAM 和 NPO + RS 更稳定；
- 平滑优化不仅能防权重层面的 relearning attack，也能一定程度上防输入层面的 jailbreak attack。

---

## 20. 技术路线总结

整篇论文的技术路线可以概括为：

```text
已有 LLM Unlearning 方法
        ↓
发现问题：模型容易被少量 forget data 重新学习恢复
        ↓
提出 Relearning Attack 鲁棒性问题
        ↓
将 Relearning Attack 建模为模型权重空间扰动
        ↓
构造 Min-Max 鲁棒 Unlearning 优化目标
        ↓
发现该目标与 SAM 形式高度一致
        ↓
将 SAM 应用于 forget loss
        ↓
训练模型在最坏权重扰动下仍然保持遗忘
        ↓
从理论上解释 SAM 与梯度惩罚、曲率平滑的关系
        ↓
扩展到 RS、GP、CR、WA 等平滑优化方法
        ↓
在 WMDP 和 MUSE benchmark 上验证抗 Relearning Attack 能力
        ↓
进一步测试 Jailbreaking Attack 下的鲁棒性
        ↓
结论：平滑 forget loss landscape 可以提升 LLM Unlearning 的鲁棒性

---
