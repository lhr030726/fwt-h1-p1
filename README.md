# FWT H1 / P1 服务器训练

本仓库提供两个主模型：**H1 确定性预测模型**，以及以本次 H1 最佳检查点为冻结骨干的 **P1 条件扩散模型**。默认分别训练 30 轮；二者都按验证集 AP 最大值选择检查点，相同 AP 保留较早轮次。这里只提供训练和验证选点，不包含测试集评估入口。

默认输入为原 RCDA 特征加 **NASA POWER 50 m 小时风的日内摘要**。这些摘要不是地面分钟阵风，也不能还原一天内每次转向。ASOS 分钟分支属于另外的可选研究方案，本仓库的默认训练未启用它。确定性模型使用既定 Focal，P1 使用既定 clean-probability BCE；扩散仍是独立主模型之一。

## 1. 登录并克隆私有仓库

服务器需有 Git 和 GitHub CLI（`gh`），登录账号必须有本私有仓库的读取权限：

```bash
gh auth login --hostname github.com
gh auth setup-git
git clone https://github.com/lhr030726/fwt-h1-p1.git
cd fwt-h1-p1
```

数据发布在同一私有仓库的 `rcda-h1-data-v1` release，数组不放进 Git。请遵守原数据使用和分享约定，不将私有 release 或下载数据改成公开资源。下载器使用现有 `gh` 登录；无人值守服务器也可通过安全的环境管理方式提供 `GH_TOKEN` 或 `GITHUB_TOKEN`，不要将令牌写入命令参数、仓库文件或日志。细粒度令牌需要此仓库的 Contents 读取权限。[GitHub release assets 文档](https://docs.github.com/en/rest/releases/assets)

## 2. 建立 Python 环境

推荐 Linux、Python 3.10、NVIDIA GPU。下面以 CUDA 11.8 的 PyTorch 2.0.0 wheel 为例，服务器驱动须支持它：

本移植包的检查包括源码与离线合成测试；尚未在目标 Linux 服务器上实测 GPU 训练。环境安装成功不等于训练链路已经验证，建议首次使用先完成后文的两轮演练。

```bash
conda create -n fwt-h1-p1 python=3.10 -y
conda activate fwt-h1-p1
python -m pip install torch==2.0.0 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
```

已有 Python 3.10 时也可先执行 `python3.10 -m venv .venv`，然后在 Linux 执行 `source .venv/bin/activate`，再执行上述两条安装命令。CPU 仅预检环境可将 PyTorch wheel 地址中的 `cu118` 换成 `cpu`；正式 GPU 训练需要 CUDA wheel。不要直接把 PyTorch 升级到最新版本，其他版本需要重新验证。[PyTorch 2.0.0 官方安装组合](https://pytorch.org/get-started/previous-versions/#v200)

## 3. 下载并核验数据

在仓库根目录执行：

```bash
python scripts/download_data.py
```

下载器先核对 release 中的 `data-index.json` 与当前 clone 内的索引字节完全一致，然后逐片下载到 `.data-cache/`，校验分片 SHA256，并逐文件校验大小和 SHA256 后安装到 `data/`。它只接受清单中的普通文件，拒绝越界路径、软硬链接、重复成员和覆盖不同内容的现有文件。索引、元数据、原始数组、天气来源文件会按训练入口约定恢复。

此版本包含 **4,576 个训练/验证样本**：Tb（确定性训练）3,197、Tr（扩散训练）811、V（验证）568。16 个压缩分片共约 **3.21 GiB**，解包后约 **13.79 GiB**；建议至少预留 **25 GiB** 空闲磁盘供下载、解包和训练输出使用。它们是完整的既定开发集；Cp、Cr 和 final 的数组不在此训练数据包内。

网络中断后可重跑同一命令：已完成且校验正确的分片会跳过，未完成的 `.partial` 分片从头下载。缓存会保留，占用空间约为压缩分片加解包数据；安装期间还需一个分片的临时空间。请勿同时运行两个下载器。若报告现有文件不符，先检查该文件来源；工具不会自动覆盖它。

只检查本地安装、无需联网或令牌：

```bash
python scripts/download_data.py --verify-only
```

需要把数据放到另一磁盘时，`--destination /mnt/fwt-project-data` 会生成 `/mnt/fwt-project-data/data/`；后续训练命令相应使用 `--data-root /mnt/fwt-project-data/data`。克隆目录中的索引仍是核验基准。

## 4. 先预检，再手动训练

下面第一条命令只做预检，不训练：

```bash
python train.py --phase det --device cuda --data-root data --run-dir runs/h1p1
```

确认预检通过、资源充足并且你决定开始正式训练后，**手动执行**以下命令。下载和预检均不会自动启动 GPU 训练：

```bash
# 先训练确定性 H1：30 轮
python train.py --phase det --device cuda --data-root data --run-dir runs/h1p1 --execute

# 等上一条成功完成，再训练 P1 扩散：30 轮
python train.py --phase diff --device cuda --data-root data --run-dir runs/h1p1 --execute
```

P1 只读取这个运行目录中本次 H1 的验证最优检查点 `runs/h1p1/formal/det/best.pth`，冻结它后训练扩散头。不要换成旧电脑上不明来源的权重。结果分别位于 `runs/h1p1/formal/det/` 和 `runs/h1p1/formal/diff/`。入口不自动重试或接着启动另一个阶段；失败后保留现场，先查看错误和运行回执。

在新服务器上只想先做两轮演练时，可分别给上述 Det、Diff 命令加 `--epochs 2`。这仍使用完整开发集，耗时取决于服务器；演练写入 `runs/h1p1/smoke-2ep/`，不会被正式扩散训练当作正式 H1 骨干使用。此选项仍需你手动加 `--execute` 才训练。

## 数据与结果边界

- 数据角色、次日标签、Tb 归一化统计和天气来源随数据包固定，不能为了指标重新划分或使用验证/测试数据重新拟合统计。
- 验证集用于选点，并报告验证集选择的阈值；本入口不会读取 held-out 测试标签进行评估，也不自动进行校准或最终测试。
- 默认 30 + 30 轮是可运行配置，不表示服务器已完成正式训练或已经证明扩散优于确定性模型。请以实际日志和完整结果为准。
- 本仓库是独立的 RCDA/FWT 实验线，不把其结果混入 WSTS 官方 12-fold 比较。

下载器离线合成测试：

```bash
python -B -m unittest discover -s tests -p 'test_download*.py' -v
```
