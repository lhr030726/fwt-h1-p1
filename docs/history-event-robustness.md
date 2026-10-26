# 历史修正 v2：本地准备，待新服务器验证

日期：2026-10-27。当前已完成本地实现、CPU 验证及小型合成 CUDA 前后向测试，
未启动真实数据正式训练。当地测试环境不替代新服务器的固定环境预检。

## 目的与证据

已完成的 v1 三种子最后五轮 AP 为 0.373151，原冻结 compact 父模型为
0.261208，同结构无历史输入对照为 0.271266。三种子均获益，但有历史的
70 个事件中 14 个退化，12 个在全部三个种子下都退化。

本地重新读取 1,379 个 Tr/V 输入火场通道，逐样本与服务器审计一致：
689 个 Tr 配对和 484 个 V 配对中，没有负变化、零变化或无效输入像元。
因此本轮不改变 signed delta，不把正负抵消、无历史回退或有效比例归一化
描述为退化原因。部分退化事件的历史天数和变化面积较少，只是事后关联；
不据此增加事件名单、剔除样本或拟合阈值。

确定的实现差异是：原训练在随机样本批次内按有效像元平均 BCE，而事件
NLL/Brier 是先把一个事件全部有效像元合并，再对事件均权。
新方案检验这种权重差异与无界修正是否影响事件稳定性；尚不声称因果确定。

## 预先固定的三组实验

| arm | 网络 | 训练损失 | 作用 |
|---|---|---|---|
| pixel_bce | 原 HistoryRefiner | 原批次有效像元 BCE | 同期基线 A |
| event_bce | 原 HistoryRefiner | 0.5 像元 BCE + 0.5 事件均权 BCE | B−A 隔离权重作用 |
| event_bce_bounded | 同参数量，修正增加平滑上界 | 同 B | C−B 隔离上界作用 |

全部使用真实历史输入。三组有相同初始参数、相同样本顺序、固定父模型、
11921 个可训练参数、零输出初始化；缺失历史或无效历史像元精确回退父模型。
新增分支从零输出重新做对照，不继承旧历史分支的 AdamW 或轮数。
原 v1 三种子的 best、last、optimizer、RNG 和完整历史保留，不能覆盖。

固定分支种子 4、17、29，每组 50 轮、2550 更新；batch 16，验证 batch 8，
AdamW lr=1e-4、weight_decay=0.01、FP32、CPU cache 16 GiB。
不同时改变数据、标签、采样、学习率、降采样比例或训练预算。
三组共享一次父模型前向，但各自计算梯度与优化器更新。

事件项只由完整 Tr 监督域计算。设 N 为 Tr 样本数、E 为有监督事件数、B 为
当前批次大小、N_e 为该事件完整 Tr 有效像元数、S_i 为样本 BCE 像元和：

```
event_bce = N / (B * E) * sum_i(S_i / N_event(i))
loss_B = 0.5 * original_batch_pixel_bce + 0.5 * event_bce
correction_C = 2.0 * tanh(raw_correction / 2.0)
```

短末批使用实际 B。不能按当前批次重新归一化事件权重，也不能叠加事件重采样。
缺历史样本仍在完整分母内；无监督事件被记录但不算入 E。
cap=2 是预先固定的候选，可能抑制有益修正，必须经过对照检验，不能称已优化。
此上界约束 logit 差值，不是把预测概率限制到 [−2,2]。

## 评估与选择

每轮计算原有全 V 指标；主汇总是每种子最后五轮平均后再跨种子汇总。
B 对 A、C 对 B 做配对比较；候选还必须相对 A 保持 pooled AP 不劣
（容差 0.001），且事件 AP、NLL、Brier、70% 召回精确率满足新训练器的
明确验收条件。报告事件改善/退化数，不能只挑最好种子或最好轮次。
检查点继续按 V pooled AP 最大、并列取最早，保留零初始化父模型回退项。
不自动替换已备份的模型，不用 final/test 选模型或阈值。

最后五轮和已选检查点输出逐事件、逐样本的修正绝对值均值/最大值、
逐事件 p95、监督像元/正例数量，以及与冻结父模型的 NLL/Brier 差值。
诊断使用 float64 从 logits 重算 proper scores，可能与原 FP32 概率指标有
极小舍入差异；它用于定位问题，不改变训练或检查点选择。
已有 history-audit 中的输入变化像元数可按 sample_id 与诊断表连接。

V 已反复用于开发，三个新分支种子也共享一个父模型；这些不构成独立泛化。
先冻结候选和阈值规则，再另行确认真正未参与调参的数据来源及事件/空间组隔离。
把已看过的 V 或旧 final 重新切分，不能重新称作盲测。

## 服务器准备与启动顺序

旧服务器已完成训练；不要再续跑旧的 50 轮任务。
使用 [已保存的备份](https://github.com/lhr030726/fwt-h1-p1/releases/tag/rescue-history-refiner-20261026)
恢复原依赖目录，并在基础包之上覆盖最终增量：

```
training-rescue-20261026T150542Z.tar.gz
weather-evidence-20261026.tar.gz
live-progress-20261026T154843Z.tar.gz
```

最终增量 SHA256：
`838c5dd482148df1f5f46ff54494736af09da13ae43c3d06a347d95c827246f3`。
保留 `/root/autodl-tmp/` 原依赖路径，用迁移包里的恢复工具上传/核对原始数组。
用户本地数据仍为 `D:\火灾数据\Wildfire_Spread_Prediction_Model\解压后\dataset`，
不需要重新预处理。保留旧模型的源码依赖目录，不能只上传一个权重文件。

新源码用独立目录 `/root/autodl-tmp/fwt-history-v2`。
使用 Git 分支在 Linux 上克隆，或使用随附 `git archive` 源码包；不要直接把
Windows 工作目录的所有文本文件用 SCP 覆盖旧目录。已确认一个历史脚本在
Windows checkout 下为 CRLF，而服务器与 Git blob 为 LF；换行差异也会破坏
冻结源码 SHA。原 35 个源码 Git blob 已与最终训练计划逐一核对一致。
原环境为 Python 3.10、PyTorch 2.7.1+cu128；其余依赖以备份的环境记录为准。
只允许新机器的 GPU 名称、Python 编译描述及文件系统 stat 身份发生变化；
Python 主/次版本、Torch 构建、数值配置、路径、大小与内容 SHA 仍须匹配。
本轮新运行会记录机器差异，旧文件不作任何修改；已产生的 v2 事务恢复仍严格
要求自己的绑定相等，不能拿 v1 transaction 冒充 v2 进度。

先运行不带执行开关的只读计划检查：

```bash
cd /root/autodl-tmp/fwt-history-v2
python tools/train_history_refiner_v2.py \
  --parent-run /root/autodl-tmp/fwt-literature-50epoch-20261026/runs/seed4-50epochs \
  --diagnostic-report /root/autodl-tmp/fwt-degradation-diagnostics-20261026/report.json \
  --experiment-plan /root/autodl-tmp/fwt-history-v2/docs/history-event-robustness.md \
  --output /root/autodl-tmp/fwt-history-v2/runs/preflight-v2
```

然后在同一命令末尾加 `--preflight`：新服务器重放全部 568 个 V 样本的
冻结父模型，验证三组零初始化输出、一次丢弃式前后向、参数隔离和显存。
这是正式训练之前的必要检查，不是性能收益证据。输出目录必须原先不存在。

只有预检通过后，才用新输出目录并附上预检来源启动：

```bash
python tools/train_history_refiner_v2.py \
  --parent-run /root/autodl-tmp/fwt-literature-50epoch-20261026/runs/seed4-50epochs \
  --diagnostic-report /root/autodl-tmp/fwt-degradation-diagnostics-20261026/report.json \
  --experiment-plan /root/autodl-tmp/fwt-history-v2/docs/history-event-robustness.md \
  --output /root/autodl-tmp/fwt-history-v2/runs/robustness-v2 \
  --preflight-run /root/autodl-tmp/fwt-history-v2/runs/preflight-v2 \
  --execute
```

程序默认不训练；正式执行必须提供与当前源码/数据/配方一致的通过预检。
遇到不一致应保留失败收据，不能通过删除哈希校验或修改原历史文件绕过。
这一版尚无新服务器 CUDA 预检、实际训练速度或效果数据。
