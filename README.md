# FWT H1 / P1：Ubuntu 22.04 从零开始训练

本公开仓库提供两个主模型：**H1 确定性预测模型**，以及以本次 H1 最佳检查点为冻结骨干的 **P1 条件扩散模型**。默认分别训练 30 轮；二者都按验证集 AP 最大值选择检查点，相同 AP 保留较早轮次。这里只提供训练和验证选点，不包含测试集评估入口。

默认输入为原 RCDA 特征加 **NASA POWER 50 m 高度小时风的日内摘要**。这些摘要不能还原每次日内转向或地面阵风。ASOS 分钟分支未在本仓库启用。H1 保留既定 Focal，P1 保留 clean-probability BCE，扩散仍是独立主模型之一。

## 1. 安装基础工具，然后匿名克隆

以下流程面向 **Ubuntu 22.04、Python 3.10、NVIDIA RTX PRO 6000 Blackwell**。无需 Miniconda，也无需 GitHub 账号或 `gh`。在服务器终端执行下面一组与你身份相符的命令：

```bash
# 当前已是 root（很多租用容器默认如此）：
apt-get update
apt-get install -y git ca-certificates python3.10 python3.10-venv
```

普通用户有 sudo 权限时改用：

```bash
sudo apt-get update
sudo apt-get install -y git ca-certificates python3.10 python3.10-venv
```

没有 root/sudo 权限时，请平台先提供这些系统工具。Ubuntu 22.04 官方软件源提供 Python 3.10 和对应的 venv 包。[Ubuntu 软件包说明](https://packages.ubuntu.com/jammy/python3.10-venv)

```bash
git clone https://github.com/lhr030726/fwt-h1-p1.git
cd fwt-h1-p1
nvidia-smi
```

`nvidia-smi` 应能显示 GPU 和驱动。如果命令不存在、显示无设备或驱动通信失败，请联系租用平台处理**宿主机驱动和容器 GPU 透传**；安装 Python 的 CUDA wheel 不能代替这一步。下面的脚本不会安装/升级 NVIDIA 驱动，也不会重启机器。

建议开始前至少准备 **40 GiB 空闲磁盘**，容纳 Python/CUDA wheel、安装临时空间、数据和初始训练输出；长期保留更多运行结果需另外增加空间。

## 2. 创建专用环境并检查 Blackwell

```bash
bash scripts/setup_ubuntu22.sh
source .venv-fwt/bin/activate
```

脚本使用系统 `/usr/bin/python3.10`，只创建仓库中的 `.venv-fwt`，通过官方 **cu128** 索引安装 **torch==2.7.1**，并安装 `requirements.txt` 固定的科学计算依赖。不安装 torchvision/torchaudio，不使用全局 pip。已有本脚本管理的环境可安全重跑；遇到没有专属标记的同名目录会停止，不覆盖或删除它。请不要在使用该环境训练期间重装环境。

这次运行时从历史的 **PyTorch 2.0.0** 迁移到 **2.7.1/cu128**，是为了支持 Blackwell：PyTorch 2.7 引入 Blackwell 支持及 CUDA 12.8 wheel，官方给出了 2.7.1/cu128 安装组合；NVIDIA 列出的 RTX PRO 6000 Blackwell 计算能力为 **12.0（sm_120）**。[PyTorch 2.7 说明](https://pytorch.org/blog/pytorch-2-7/)、[2.7.1 官方安装组合](https://pytorch.org/get-started/previous-versions/#v271)、[NVIDIA GPU 计算能力](https://developer.nvidia.com/cuda/gpus)

安装后脚本执行 `pip check`，记录 Python/依赖/驱动/GPU 能力，并在选中的 GPU 上检查极小的 FP32 矩阵乘法和卷积。这不是模型训练，不读取数据或权重。检查结果写入 `runs/environment/`；进入环境检查后的失败也保留快照。更早的系统检查或 pip 安装失败会直接在终端报错。默认检查可见 GPU 0；多卡机器可用 `bash scripts/setup_ubuntu22.sh 1` 指定可见 GPU 1，或先设置 `CUDA_VISIBLE_DEVICES` 再使用默认 0。

单独复查环境的命令：

```bash
# 只记录版本与依赖兼容性，不导入 torch、不初始化 GPU：
python scripts/check_environment.py

# 明确执行一次小型 GPU 运算检查：
python scripts/check_environment.py --gpu-test --device-index 0
```

模型代码、损失、数据划分和归一化配方未随这次环境迁移改变。**历史 GPU 小试是在旧运行时完成的；尚未证明 2.7.1 与 2.0.0 数值等价，也未在你的 Ubuntu/PRO 6000 上实测完整模型训练。**环境检查通过只说明该机器能执行所检查的操作，建议先手动做后文的两轮演练。

## 3. 匿名下载并核验数据

在仓库根目录、激活环境后执行：

```bash
python scripts/download_data.py
```

数据位于本仓库的 `rcda-h1-data-v1` release，数组不放进 Git。公开下载不要求登录或令牌；若 GitHub 匿名 API 限流，可等待限流恢复，或按下载器提示安全地使用已有登录/环境令牌，不要把令牌写入仓库文件或命令参数。数据使用与来源说明见 [NOTICE](third_party/NOTICE.md)。

下载器先核对 release 的 `data-index.json` 与当前 clone 中索引的原始字节一致，再逐片下载到 `.data-cache/`。所有压缩片及内部文件都核验大小和 SHA256，随后安装到 `data/`；不接受越界路径、链接、重复成员或覆盖不同内容的现有文件。

数据包含 **4,576 个完整开发样本**：Tb（H1 训练）3,197、Tr（P1 训练）811、V（验证）568。16 个分片压缩后约 **3.21 GiB**，解包后约 **13.79 GiB**。Cp、Cr 和 final 的数组不在此包中。

网络中断后重跑相同命令：已完成且校验正确的分片会跳过，未完成分片从头下载。缓存保留；安装时还需一个分片的临时空间。不要同时运行两个下载器。只核验本地安装可以执行：

```bash
python scripts/download_data.py --verify-only
```

放到其他磁盘时，可用 `--destination /mnt/fwt-project-data`；它会创建 `/mnt/fwt-project-data/data/`，后续训练相应设置 `--data-root /mnt/fwt-project-data/data`。

## 4. 预检，再手动运行 H1 和 P1

这条命令只检查数据、配置与来源绑定，**不训练、不初始化 GPU**：

```bash
python train.py --phase det --device cuda --data-root data --run-dir runs/h1p1
```

首次上新服务器，建议先手动执行两轮演练：

```bash
python train.py --phase det --device cuda --data-root data --run-dir runs/h1p1 --epochs 2 --execute
# 等 H1 成功完成后再执行：
python train.py --phase diff --device cuda --data-root data --run-dir runs/h1p1 --epochs 2 --execute
```

**两轮演练仍使用完整的 Tb/Tr/V，不是只跑几个 batch，可能需要较长时间。**输出位于 `runs/h1p1/smoke-2ep/`，不会被正式 P1 当作正式 H1 骨干使用。演练完成后检查实际日志，再决定是否启动正式预算。

正式训练由你分别手动启动；下面每个阶段默认 30 轮：

```bash
python train.py --phase det --device cuda --data-root data --run-dir runs/h1p1 --execute
# 等本次正式 H1 成功完成后再执行：
python train.py --phase diff --device cuda --data-root data --run-dir runs/h1p1 --execute
```

P1 只读取同一运行的 `runs/h1p1/formal/det/best.pth`，完成校验后冻结该骨干，再训练扩散头。结果分别位于 `runs/h1p1/formal/det/` 和 `runs/h1p1/formal/diff/`。入口不自动重试或串联下一阶段；已有非空输出不会被覆盖。新实验请选择新的 `--run-dir`，失败则保留现场并先检查回执。

## 可选的资源利用率实验

此分支提供 CPU 样本缓存和显式批量参数。默认仍是训练 `batch_size=1`、验证批量跟随训练批量、样本缓存关闭；地理基底缓存保留最多 1,024 个网格。它们是待实测的性能设置，不表示预测质量已经改善。

`--sample-cache-gib 40` 让训练与验证共享一个最多保留 40 GiB 张量存储的 CPU 缓存。首次取样仍走原始 Dataset 读取与派生计算，天气来源验证在 Dataset 构造时完成；后续命中复用确定性计算结果，返回值独立复制，避免调用者修改缓存。当前完整开发数据的张量净大小估计约 33.52 GiB，实际进程内存还包括地理基底、临时张量、验证预测及分配器开销。缓存不是总内存上限，保持 `num_workers=0`，也不会主动填满显存。每轮的 `diagnostics.json` 会记录缓存命中、逐出和净字节。

仅启用缓存、保持 batch=1 的示例（先完成新环境演练与性能/等价性检查，再使用全新输出目录）：

```bash
python train.py --phase det --device cuda --data-root data \
  --run-dir runs/h1p1-cached --sample-cache-gib 40 --execute
# 等该 H1 完成后，以相同资源设置启动它自己的 P1：
python train.py --phase diff --device cuda --data-root data \
  --run-dir runs/h1p1-cached --sample-cache-gib 40 --execute
```

`--batch-size N` 是训练配方变体：它改变每轮优化器更新次数和梯度聚合，不能仅因显卡空闲就视作原 batch=1 配方的等价加速。`--validation-batch-size N` 可单独设置验证批量；需核对预测与指标的数值差异。两项批量均为 1–128，缓存预算为 0–64 GiB。实际设置写入预检、检查点和完成回执，H1/P1 的绑定必须一致；不能用改过设置的新 P1 冒充原 H1 配套运行。

用 `scripts/benchmark_geography_cache.py` 核验地理缓存输出与 CPU 时间；`scripts/benchmark_batches.py` 默认只输出计划，显式执行时要求 GPU 空闲，测量的是合成输入下的计算容量。正式采用配置还应比较实际数据的稳定 samples/s、峰值内存、梯度和完整验证表现。原始数组、归一化、天气来源核验和每轮输入/源码检查仍保留。

## 数据与结果边界

- 次日目标、事件角色、Tb 归一化及天气来源固定；不使用验证/测试数据重新拟合统计或重新划分数据。
- V 用于最大 AP 选点及阈值选择；本入口不自动进行独立校准或最终测试。
- 30 + 30 是待用户执行的配置，不表示已经完成长训练或已经证明扩散优于 H1。
- 这是独立 RCDA/FWT 实验线，不将其结果混入 WSTS 官方 12-fold 比较。

已完成检查及未验证范围见 [VALIDATION.md](VALIDATION.md)。离线合成测试可运行 `python -B -m unittest discover -s tests -v`；它们不等于你的实际 GPU 模型训练验收。
