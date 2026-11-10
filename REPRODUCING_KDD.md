# AlphaGen KDD 沪深 300 复现

上游基线是 `e3b152669c17f481eafabed9cf7d3ad583eaa84f`。本地运行环境使用 Windows x64、Python 3.8.20 和 PyTorch 2.0.1 CPU。Qlib 使用 `pyqlib==0.9.6`，其 Windows 扩展需要比上游要求更新的 NumPy。PPO 和 Gym 保留同代 API，采用 `stable-baselines3==1.8.0`、`sb3-contrib==1.8.0` 和 `gym==0.21.0`。

## 安装

从 AlphaGen 仓库根目录在 PowerShell 中运行。锁文件固定本次实测环境的包版本，Gym 另从 PyPI 官方源码包安装，因为其旧版元数据无法通过现代解析器的校验。

```powershell
$env:UV_CACHE_DIR = Join-Path $PWD '.uv-cache'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $PWD '.python'
uv python install 3.8.20 --no-bin
uv venv --python 3.8.20 .venv
uv pip install --python .venv\Scripts\python.exe --no-deps -r requirements-kdd-windows.lock.txt
$archive = Join-Path $env:UV_CACHE_DIR 'gym-0.21.0.tar.gz'
curl.exe --fail --location 'https://files.pythonhosted.org/packages/4b/48/920cea66177b865663fde5a9390a59de0ef3b642ad98106ac1d8717d7005/gym-0.21.0.tar.gz' --output $archive
if ($LASTEXITCODE -ne 0) { throw 'Gym download failed' }
if ((Get-FileHash $archive).Hash -ne '0fd1ce165c754b4017e37a617b097c032b8c3feb8a0394ccc8777c7c50dddff3') { throw 'Gym archive hash mismatch' }
.\.venv\Scripts\python.exe -m pip install --no-index --no-deps --no-build-isolation $archive
```

## 数据和短跑

数据构建读取 Tushare 的未复权日线、复权因子和交易日历。使用截至 `2023-01-15` 的复权因子统一前复权价格，反向调整成交量并计算 VWAP。沪深 300 成分优先从有生效区间的 Qlib 格式文件导入。若使用 `--monthly-members`，则从 Tushare `index_weight` 取各月最新快照，从快照日期后的首个交易日开始生效。这是成分区间近似，快照清单会记录来源。下载缓存和快照保存在 `data/`。

本次使用公开调样记录重建的 2008-06-01 至 2022-03-31 成分区间。重建以 2009-11-30 的 Tushare 300 只成分快照为锚点，按调样生效日向前和向后推算。每个区间与下方固定提交发布的区间逐个变更日核对，成分集合一致。转换为 Qlib 格式时，将排他结束日减一天，按所需范围裁剪，代码补 `SH` 或 `SZ` 前缀。[调样记录和区间](https://github.com/dongzhaohe321418-lab/csi300-spmo-index/tree/c85b24f1f6ed69e8b615ebbe42f7e66b29b4c830/data/processed)

Tushare 对 `600849.SH` 返回空行情，已核实 `601607.SH` 下有更名前的历史记录。快照保留旧代码的成分区间，并在清单中记录行情来源映射。[上海医药 2010 年年报](https://static.sse.com.cn/cs/zhs/scfw/gg/ssgs/2011-03-09/601607_2010_n.pdf)

```powershell
$env:TUSHARE_TOKEN = [Environment]::GetEnvironmentVariable('TUSHARE_TOKEN', 'User')
.\.venv\Scripts\python.exe -m data_collection.fetch_tushare_data --probe
.\.venv\Scripts\python.exe -m data_collection.fetch_tushare_data --members data\csi300.txt --output data\tushare-csi300

$env:ALPHAGEN_QLIB_DIR = Join-Path $PWD 'data\tushare-csi300'
$env:MPLCONFIGDIR = Join-Path $PWD 'out\mpl'
$env:MPLBACKEND = 'Agg'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
.\.venv\Scripts\python.exe train_maskable_ppo.py --seed 0 --pool-capacity 10 --steps 2048 --device cpu --output-dir out\smoke
```

短跑输出包括模型、因子池 JSON 和 2019 年的验证指标 JSON。训练与验证切分各剔除末尾 20 个交易日的特征行，使前瞻收益标签留在各自的切分内。数据目录仍保留完整时间范围及前后缓冲。

## 正式复现口径

目标是 KDD 2023 论文 Table 2 的 `Ours*` 沪深 300 两列。使用种子 0–9。池容量 10、20、50、100 的训练预算分别为 250,000、300,000、350,000、400,000 步，取自 KDD 冻结提交中的预算字典。PPO 按完整 rollout 执行，因此实际步数向上取整到 2048 的倍数。[论文 Table 2](https://arxiv.org/pdf/2306.12964#page=7) · [冻结版预算](https://github.com/ICT-FinD-Lab/alphagen/blob/e3b1526/train_maskable_ppo.py)

每个容量和种子保存验证 IC 最高的检查点，同分保留较早的检查点。用十个种子的平均验证 IC 选容量，同分取较小容量。随后只评估所选容量的十个因子池，输出测试集 IC 和 Rank IC 的均值、样本标准差及每个种子的结果。原论文的检查点和最优容量选择规则未能核实，这里使用预先固定的验证集规则。

准备好本地快照后，从 AlphaGen 仓库根目录运行。使用上节的 `MPLCONFIGDIR`、`MPLBACKEND`、`OMP_NUM_THREADS` 和 `MKL_NUM_THREADS` 环境设置。

```powershell
.\.venv\Scripts\python.exe reproduce_table2.py train --config reproduction-table2.json --output out\table2-csi300
.\.venv\Scripts\python.exe reproduce_table2.py test --config reproduction-table2.json --output out\table2-csi300
```

训练命令可从完成的容量与种子组合继续。评估命令读取冻结配置和验证检查点，写入 `out/table2-csi300/table2-csi300.json`。
