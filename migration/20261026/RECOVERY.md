# 服务器到期迁移与恢复

本次备份保存源码、训练日志、完整成对事务检查点、各分支最佳和最后权重、AdamW 动量和步数、随机数状态、冻结 H1 与 compact best49，以及验证它们需要的历史文件。GitHub 保存这些文件不会自动提供 GPU 或继续运行 Python 进程。

## 恢复材料

- `training-rescue-20261026T150542Z.tar.gz`：训练及依赖闭包，内部 `payload/root/autodl-tmp/` 对应原服务器目录。
- `weather-evidence-20261026.tar.gz`：5,930 个气象来源证据文件，未压缩 102,694,064 字节。它们是运行依赖，不能只复制 `weather_manifest.json`。
- `latest-progress-20261026T151152Z.tar.gz`：暂停训练后最后一致的轮次与事务，必须在基础训练包之后合并同名路径，以增量版本为准。
- `environment-and-weather.json`：实际环境包版本与系统信息。
- `restore_training_arrays.py`、`resume_on_new_server.py`：恢复数据与跨机器续训入口。
- `local-data-full-verification.json`：用户本地数据与原服务器 Tr/V 数组全部哈希一致的检查结果。

下载后先用发布页面附带的 SHA256 清单检查压缩包；再用包内 `snapshot-manifest.json` 核对文件。原目录包含多层历史实验，因为冻结校验会读取历史收据与检查点；不能只拿最新的 `best.pth`。

## 数据

原始数据仍在用户本地：

```text
D:\火灾数据\Wildfire_Spread_Prediction_Model\解压后\dataset
```

已完整核对当前训练所需的 **2,758 个输入/标签文件，共 4,428,686,080 字节**，全部 SHA256 与服务器绑定一致，不需要重新预处理。把该数据集上传或挂载至新服务器后，运行：

```bash
python restore_training_arrays.py \
  --dataset /mnt/original-dataset \
  --index /root/autodl-tmp/fwt-h1-p1/data/arrays_index.json \
  --data-root /root/autodl-tmp/fwt-h1-p1/data \
  --receipt /root/autodl-tmp/restored-arrays-receipt.json
```

它只复制原始字节并核对 SHA256，遇到不同内容的已有目标会停止。默认恢复 `residual_train` 与 `validation`；需要重训 H1 时才加 `--all-development`。不要把 `arrays/` 简单做成指向其他位置的符号链接，因为严格绑定使用解析后的绝对路径。

## 新 Linux GPU 服务器

在全新或空目录中还原压缩包中的原绝对目录，保留原始备份不改。例如先将基础训练包解压至临时目录，将最终增量包解压并合并到同一临时目录，再把最终 `payload/root/autodl-tmp/` 内容复制到 `/root/autodl-tmp/`。基础包清单是 `snapshot-manifest.json`，最终增量清单是 `latest-progress-manifest.json`；先分别验证再合并。气象包的归档根目录是 `root/autodl-tmp/`，没有 `payload/` 这一层，解压到 `/` 后才是正确的 `/root/autodl-tmp/...`。新服务器已有同名文件应先核对哈希，不覆盖无关运行。

环境使用 Python 3.10，PyTorch **2.7.1+cu128**，NumPy 1.22.3、SciPy 1.11.4、scikit-learn 1.2.2、pyproj 3.7.1；其他实际版本见环境记录。新服务器需要能运行 CUDA 12.8 版 PyTorch 的 NVIDIA 驱动，以及恰好一个可见 GPU。原过程 FP32、batch 16、CPU 样本缓存上限 16 GiB；当前设置保持不变。

冻结代码在：

```text
/root/autodl-tmp/fwt-history-refiner-20261026-r3
```

先执行恢复预检查，不带 `--execute`：

```bash
python resume_on_new_server.py \
  --repo /root/autodl-tmp/fwt-history-refiner-20261026-r3 -- \
  --parent-run /root/autodl-tmp/fwt-literature-50epoch-20261026/runs/seed4-50epochs \
  --diagnostic-report /root/autodl-tmp/fwt-degradation-diagnostics-20261026/report.json \
  --experiment-plan /root/autodl-tmp/fwt-history-refiner-20261026-r3/history-plan.md \
  --resume-run /root/autodl-tmp/fwt-history-refiner-20261026-r3/runs/history-paired \
  --output /root/autodl-tmp/fwt-history-resumed/runs/history-paired
```

通过后在同一命令末尾添加 `--execute` 正式继续。输出必须是尚不存在的新目录。预算仍是三个种子、每种子两个分支各 50 轮；已完成轮次不会重复计入，也不会自动扩为 100 轮。恢复采用每个种子的原子 `transaction.pth`，所以进度以完整提交的轮次为准；断电时尚未提交的半轮会重跑。

最后备份的两分支共同进度：**种子 4 已完成 50/50；种子 17 已完成 50/50；种子 29 已完成 35/50**。恢复后只有种子 29 从第 36 轮继续；已完成种子可能重新验证/生成输出收据，但不会再次进行训练更新。原始已完成种子的收据保留在备份中。

## 跨机器恢复校验的范围

原训练器会把 inode、设备号、mtime、Python 编译字符串、GPU 名称一起作严格相等比较，因此把文件原样复制到另一台机器也会被拒绝。`resume_on_new_server.py` 是单独记录哈希的迁移适配器：

1. 只接受冻结训练器源码 SHA256，保留原始源码、JSON、权重和历史收据不改。
2. 保留所有文件内容哈希、样本顺序、历史配对、模型、损失、优化器与训练预算检查。
3. 只允许文件系统身份和机器描述变更；文件路径、大小、样本身份、Torch 版本及数值设置仍须一致。
4. 在内存中产生包含新机器绑定的事务，逐个验证模型、AdamW、RNG、计数和历史序列没有变化；新输出旁保存独立 `*.machine-relocation.json`。
5. 完整重放冻结父模型验证集，继续执行原 AP/指标容差检查，未通过则停止。

适配器已经通过 **3 项 CPU 测试**，覆盖身份差异、拒绝配方/依赖变化、训练过的事务状态精确保留。**尚未在另一台 GPU 服务器上进行真实恢复训练**；不同 GPU 不保证浮点结果逐位一致，原父模型重放检查仍可能要求进一步处理。它不会静默降低指标校验。

原训练文件必须保持冻结。若以后迁移再次恢复，使用最新续训目录作为 `--resume-run`，仍保留原 parent-run 与历史依赖。
