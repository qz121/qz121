## Unlearning Baseline 对比表


| Baseline | 全称 / 类型 | 核心做法 | 直观理解 | 优点 | 缺点 | 典型失败模式 |
|---|---|---|---|---|---|---|
| GA | Gradient Ascent / Gradient Difference | 在 **forget set** 上增大原答案的损失，让模型“学坏”这部分记忆；有时同时在 **retain set** 上做正常训练 | 像用橡皮擦硬擦掉一段记忆 | 简单直接、实现容易、遗忘效果通常很强 | 容易过度破坏模型，把 retain knowledge 也一起伤到 | 输出乱码、模型崩坏、retain set 性能大幅下降 |
| KL | KL Minimization | 在 forget set 上做遗忘，同时约束 unlearn 后模型在 **retain set** 上的输出分布不要偏离原模型太远 | 一边删记忆，一边给模型拴一根“别跑太远”的绳子 | 比纯 GA 更稳定，能一定程度保留原模型行为 | 仍可能出现灾难性遗忘；有时只是“稍微稳一点的 GA” | retain set 依然掉点明显，甚至生成无意义文本 |
| IDK Tuning | Refusal Supervision | 把 forget set 的原始答案替换成 “I don’t know.” 这类拒答语句，再做微调 | 不是把模型打坏，而是教它“学会闭嘴” | 实现简单；输出通常比 GA/KL 更自然 | 容易把“拒答”泛化到不该拒答的问题上 | 对普通问题也开始拒答，出现 over-refusal |
| PO | Preference Optimization | 把 unlearning 写成偏好学习：让模型更偏好“拒答/安全回答”，而不是原始敏感答案 | 训练模型“更喜欢拒答，不喜欢泄露” | 行为上更自然，适合和对齐方法结合 | 需要构造 preference pair；也可能把拒答学过头 | 模型变得过于保守，很多正常问题也拒答 |
| MANU | Pruning-based Unlearning | 找出对 forget set 最重要的神经元，然后剪掉 | 不是重新训练，而是直接拆掉一部分“存记忆的零件” | 思路直观，不完全依赖 loss 设计 | 剪枝边界难控制，容易误删有用能力 | 回答错误、语言混乱、retain set 也受损 |

---

- **GA**：硬删记忆  
- **KL**：硬删记忆，但尽量别偏离原模型太远  
- **IDK Tuning**：把敏感答案统一教成“我不知道”  
- **PO**：让模型偏好“拒答”而不是“泄露”  
- **MANU**：直接剪掉最像在存这段知识的神经元  

---


几类经典的 unlearning baseline 作为对比，包括：  
**(1) 基于梯度反向优化的方法**，如 GA；  
**(2) 基于分布约束的方法**，如 KL Minimization；  
**(3) 基于拒答监督的方法**，如 IDK Tuning；  
**(4) 基于偏好优化的方法**，如 PO；  
**(5) 基于结构剪枝的方法**，如 MANU。

---
