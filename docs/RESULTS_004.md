# 实验 004 完整结果

2026-09-23 完成。仅 seed404；silver 开发验证，验证集参与选型，不是独立测试。所有数据来自公开包内 reports/published 的机器可读汇总。

## 效果：288 截图 / 1314 判断

| 权重 | 方法 | 准确率 | Macro-F1 | 格式有效率 | 误报成功率 |
|---|---|---:|---:|---:|---:|
| base-full | json | 36.53% | 31.41% | 74.31% | 56.20% |
| base-full | json_constrained | 49.24% | 39.83% | 100.00% | 59.43% |
| base-full | logits | 48.71% | 33.03% | 100.00% | 68.35% |
| base-full | fast | 48.93% | 33.30% | 100.00% | 67.57% |
| decision-selected-full | json | 30.52% | 39.18% | 52.08% | 24.42% |
| decision-selected-full | json_constrained | 60.73% | 60.48% | 100.00% | 39.28% |
| decision-selected-full | logits | 90.41% | 92.06% | 100.00% | 5.68% |
| decision-selected-full | fast | 90.18% | 91.88% | 100.00% | 5.56% |
| json-selected-full | json | 89.80% | 92.17% | 97.92% | 6.59% |
| json-selected-full | json_constrained | 91.25% | 92.67% | 100.00% | 7.24% |
| json-selected-full | logits | 78.01% | 73.68% | 100.00% | 15.25% |
| json-selected-full | fast | 78.16% | 73.48% | 100.00% | 15.12% |

## 延迟：每 K / 方法 100 图 × 5 次

单位 ms；p50/p95 为汇总实现的经验分位数，可能与插值分位数有小数差异。格式有效率仅针对测速子集，不能与全量效果列混用。

| 权重 | K | 方法 | p50 | p95 | 格式有效率 |
|---|---:|---|---:|---:|---:|
| base-full | 1 | json | 141.24 | 247.09 | 0.0% |
| base-full | 1 | json_constrained | 133.76 | 175.97 | 100.0% |
| base-full | 1 | logits | 59.85 | 100.57 | 100.0% |
| base-full | 1 | fast | 59.02 | 98.21 | 100.0% |
| base-full | 2 | json | 314.39 | 648.12 | 25.0% |
| base-full | 2 | json_constrained | 178.79 | 204.59 | 100.0% |
| base-full | 2 | logits | 122.03 | 200.54 | 100.0% |
| base-full | 2 | fast | 105.93 | 182.67 | 100.0% |
| base-full | 4 | json | 520.45 | 914.95 | 70.0% |
| base-full | 4 | json_constrained | 250.81 | 268.78 | 100.0% |
| base-full | 4 | logits | 239.32 | 314.98 | 100.0% |
| base-full | 4 | fast | 194.57 | 286.81 | 100.0% |
| decision-selected-full | 1 | json | 88.95 | 123.34 | 0.0% |
| decision-selected-full | 1 | json_constrained | 128.05 | 163.91 | 100.0% |
| decision-selected-full | 1 | logits | 58.73 | 92.48 | 100.0% |
| decision-selected-full | 1 | fast | 58.93 | 94.65 | 100.0% |
| decision-selected-full | 2 | json | 124.87 | 202.09 | 37.0% |
| decision-selected-full | 2 | json_constrained | 165.42 | 190.86 | 100.0% |
| decision-selected-full | 2 | logits | 81.77 | 134.78 | 100.0% |
| decision-selected-full | 2 | fast | 70.23 | 122.25 | 100.0% |
| decision-selected-full | 4 | json | 293.12 | 315.53 | 57.0% |
| decision-selected-full | 4 | json_constrained | 242.85 | 257.51 | 100.0% |
| decision-selected-full | 4 | logits | 233.05 | 312.73 | 100.0% |
| decision-selected-full | 4 | fast | 185.81 | 270.12 | 100.0% |
| json-selected-full | 1 | json | 137.31 | 170.61 | 100.0% |
| json-selected-full | 1 | json_constrained | 139.26 | 173.86 | 100.0% |
| json-selected-full | 1 | logits | 58.09 | 108.02 | 100.0% |
| json-selected-full | 1 | fast | 57.29 | 102.68 | 100.0% |
| json-selected-full | 2 | json | 177.45 | 203.43 | 100.0% |
| json-selected-full | 2 | json_constrained | 179.37 | 214.18 | 100.0% |
| json-selected-full | 2 | logits | 127.34 | 211.36 | 100.0% |
| json-selected-full | 2 | fast | 110.57 | 191.94 | 100.0% |
| json-selected-full | 4 | json | 243.13 | 264.08 | 100.0% |
| json-selected-full | 4 | json_constrained | 246.91 | 268.52 | 100.0% |
| json-selected-full | 4 | logits | 252.34 | 329.40 | 100.0% |
| json-selected-full | 4 | fast | 214.33 | 289.29 | 100.0% |

## 如何解释

- 原始快速 logits 到决策快速 logits：48.93% → 90.18%，同推理协议，支持本开发分布上训练有效。
- 决策快速 logits 对 JSON SFT 普通生成：90.18% 对 89.80%，差异+0.38个百分点；group bootstrap 95%区间[-2.10,+3.05]，不能证明质量优势。JSON SFT约束生成91.25%，比决策快速路径更高。
- 同决策权重的约束 JSON / fast p50 比：K1 2.17×、K2 2.36×、K4 1.31×。同一权重两种服务协议不同，上下文不完全等价。
- 不同训练系统 JSON SFT普通 / 决策fast p50比：2.33× / 2.53× / 1.31×。不是纯推理结构的因果贡献。
- K4决策fast p95 270.12ms，高于同决策权重约束JSON 257.51ms，不能声称所有尾延迟更快。
- JSON SFT权重直接读logits约78%并不等同决策LoRA的90%；训练目标与推理协议必须配套。
- 校准T=2.14355；验证ECE10从0.06665到0.02935、NLL从0.31690到0.23604，三分类Brier平方和从0.15826到0.13721。未接入默认服务、未验证生产阈值。
- bootstrap以269个group聚类、2000重复；区间以已选权重为条件，不计入选型偏差与教师标注偏差。
- 不提供K8、并发4/8或多seed结果，因为尚未执行。

## 本地证据归档

远端结果归档包含110个JSON/JSONL文件，下载后SHA-256与远端一致：`dde16e8f50f118a51c2fecf6c63e48b52e938d8e11f219b5c5d7c4d6c3565cc7`。公开包只包含去除业务原文及基础设施信息的汇总，不含该内部完整归档。
