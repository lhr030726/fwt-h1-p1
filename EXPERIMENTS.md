# 新增实验记录

历史结果以此前运行收据和 `rescue-history-refiner-20261026` 备份为准。
本文件从本次新增候选开始记录，不能视作旧实验结果的替代。

| 项目 | history-event-robustness-v2 |
|---|---|
| 日期 / 状态 | 2026-10-27；本地开发完成，等待新 GPU 服务器，未正式训练 |
| 协议 | 保留既有 residual_train / validation；不是另起官方 12-fold 实验 |
| 输入 | 原特征 + 严格同事件/网格的前一天火场变化；不改变标签或归一化 |
| 模型 | 冻结 H1 + compact best49 + HistoryRefiner；第三组增加平滑修正上界 |
| 三组 | pixel_bce / event_bce / event_bce_bounded |
| 参数量 | 各组新增可训练参数 11921；父模型不更新 |
| 损失 | A 原像元 BCE；B/C 固定0.5像元 BCE+0.5完整Tr事件均权 BCE |
| 修正上界 | C: 2*tanh(raw_delta/2)，待检验，不宣称最优 |
| 优化器 | AdamW，lr=1e-4，weight_decay=0.01，无 scheduler/EMA |
| batch / 精度 | 训练16，验证8，FP32，cache16GiB |
| 种子 / 预算 | 4、17、29；每组50轮、2550更新；等待运行 |
| 检查点规则 | V pooled AP最大，并列最早，保留零初始化父模型回退 |
| 比较规则 | 每种子最后5轮均值；B−A、C−B、C−A；详见预先固定计划 |
| 本地验证 | 54个不同功能测试通过，含新三组CPU/CUDA合成集成；不等价于真实数据GPU预检 |
| 独立审查 | 事件损失公式、mask、迁移字段、预检、事务恢复已只读审查；诊断标签校验已修复 |
| 效果 / AP / F1 / IoU / precision / recall | 待正式训练及验证，未产生 |
| GPU显存 / 速度 / 正式best checkpoint | 待新服务器测量，未产生 |
| 独立测试 | 尚未进行；现有V和已看过的旧final不能称新盲测 |
| Git来源 | 模块e56b5de；训练器ae60425；完整提交以当前分支及归档清单为准 |

[冻结候选计划与启动步骤](docs/history-event-robustness.md) · [本地退化诊断摘要](docs/history-event-diagnosis.md)
