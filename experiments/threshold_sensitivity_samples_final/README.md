# 阈值敏感性实验样本集

本目录保存传统模式阈值敏感性实验使用的样本文档对。样本由 `data/` 目录中的论文文档抽取并构造，分为直接复制、轻度改写、同主题不抄袭和完全无关四类。

## 文件说明

- `samples/targets/`：待检测文本。
- `samples/references/`：参考文本。
- `labels.csv`：样本标签、来源文档、样本类型和长度信息。
- `labels.jsonl`：与 `labels.csv` 等价的 JSON Lines 格式。
- `threshold_grid.csv`：建议测试的阈值组合。
- `source_manifest.csv`：成功解析并用于样本构造的源文档清单。

## 标签含义

- `label = 1`：应判为相似。
- `label = 0`：不应判为相似。

## 样本构成

样本总数：30

| sample_type | count |
|---|---:|
| direct_copy | 8 |
| light_rewrite | 8 |
| same_topic_non_plagiarism | 8 |
| unrelated | 6 |

| label | count |
|---|---:|
| 0 | 14 |
| 1 | 16 |

## 建议实验流程

1. 第一轮固定细粒度阈值为 `0.30`，测试 `threshold_grid.csv` 中的窗口阈值。
2. 根据 Precision、Recall、F1、FP、FN 选择较均衡的窗口阈值。
3. 第二轮固定选出的窗口阈值，再测试细粒度确认阈值。
4. 使用 `labels.csv` 作为真实标签表。

第一次统计时，可以先把 `hit_count > 0` 定义为系统检出；如果后续需要更严格，可以改成 `hit_count > 0 and score >= fine_threshold`。
