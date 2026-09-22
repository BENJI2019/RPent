---
name: local-vs-mtp
description: Explains local single-GPU vs MTP multi-GPU jobs on this cluster: required env preamble, dataset/model/algorithm path mappings, and how local vs MTP introduce the environment. Use when writing or reviewing MTP launchers, comparing 单卡 and 多卡, mapping /home/ma-user/work/algorithm to /opt/huawei/schedule-train/algorithm, or setting CUDA/NCCL/conda on MTP.
---

# 本机单卡 vs MTP 多卡

MTP 由人在平台上提交，本机只跑单卡/少卡调试。多卡脚本开头必须先做软链和下面这套环境，再 `conda activate`，再跑业务。

## 算法包（一次任务只能挂一个）

平台把**当前提交的那一个**算法包摊平到 `schedule-train/algorithm/`，不会带上本机的包名。

| | 路径 |
|---|---|
| 本机 | `/home/ma-user/work/algorithm/<算法包>/<相对路径>` |
| MTP | `/opt/huawei/schedule-train/algorithm/<相对路径>` |

例：本机 `/home/ma-user/work/algorithm/inference_llb_all/foo/run.sh`  
→ MTP `/opt/huawei/schedule-train/algorithm/foo/run.sh`

写多卡脚本时：

- 不要写死 `/home/ma-user/work/algorithm/<算法包>/...`，平台上没有这一层包名。
- 用 `SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"` 找本包内文件。
- 不要引用本机 `algorithm/` 下**另一个包**的路径，这次任务里不存在。
- 改了本机源码必须重新提交，集群跑的是提交时的快照。

## 存储路径对应

MTP 只挂 `/opt/huawei/{dataset,quoteModel,schedule-train}`。脚本里继续写 `/home/ma-user/work/...`，靠启动时软链对齐。

| 逻辑路径（脚本里写这个） | MTP 真实盘 |
|---|---|
| `/home/ma-user/work/dataset` | `/opt/huawei/dataset` |
| `/home/ma-user/work/model` | `/opt/huawei/quoteModel` |
| `/opt/huawei/explorer-env/dataset` | `/opt/huawei/dataset` |
| `/opt/huawei/explorer-env/quoteModel` | `/opt/huawei/quoteModel` |

本机方向常相反：`/opt/huawei/dataset` → `/home/ma-user/work/dataset`。本机不必、也不要重做 MTP 那组 `ln -s`（会打到已有目录上）。

权重、数据、图片都走 `dataset` / `model` 这两套盘，不要写节点本地盘。

## 本机和 MTP 怎么引入环境

| | 本机单卡 | MTP 多卡 |
|---|---|---|
| 磁盘 | `work/{algorithm,dataset,model}` 已经在 | 只有 `/opt/huawei/*`，脚本自己软链 |
| 算法树 | 带 `<算法包>` 这一层 | 包根 = `schedule-train/algorithm/` |
| Python | 本机已 activate 的那个环境 | 先 `conda init` + `source conda.sh`，再 activate **同一环境的绝对路径**（`echo $CONDA_PREFIX`） |
| CUDA/NCCL | 本机环境往往已够用 | 必须导出下面这套 `CUDA_HOME` / NCCL；activate 后再导 CUDA，cuDNN 跟 `$CONDA_PREFIX` |
| 分布式 | 不用 Ray/多机 NCCL | 读 `MA_NUM_HOSTS` `MA_NUM_GPUS` `VC_TASK_INDEX` `VC_WORKER_HOSTS` |
| 启动 | 小流量、短步数、独立输出 | 平台填多卡脚本；改代码后重新提交 |

本机误跑多卡脚本：没注入 `MA_NUM_GPUS` 时，很多脚本会按 **8 卡** 去拉。

## 多卡脚本必须先写的环境（按顺序）

下面这段是通用前缀，业务命令放在最后。

### 1. 软链

```bash
mkdir -p /opt/huawei/explorer-env/
ln -s /opt/huawei/dataset /opt/huawei/explorer-env/dataset
mkdir -p /home/ma-user/work/
ln -s /opt/huawei/dataset /home/ma-user/work/dataset
ln -s /opt/huawei/quoteModel /home/ma-user/work/model
```

让 `/home/ma-user/work/{dataset,model}` 和 `explorer-env/dataset` 指到平台盘。很多工具只认 `explorer-env`。`ln -s` 已存在会失败，可改 `ln -sfn` 或 `2>/dev/null || true`。

### 2. Python 3.12 distutils

```bash
export SETUPTOOLS_USE_DISTUTILS=local
```

3.12 没有标准库 `distutils`。不设时 Ray / setuptools 可能走 `stdlib`，`import` 直接挂。

### 3. CUDA 12.8

```bash
export CUDA_HOME=/opt/huawei/dataset/Common_wl/pkgs_llb/cuda-12.8
export PATH=${CUDA_HOME}/bin:$PATH
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH
export CUDACXX=${CUDA_HOME}/bin/nvcc
export CUDAToolkit_ROOT=${CUDA_HOME}
export CUDA_TOOLKIT_ROOT_DIR=${CUDA_HOME}
```

钉死共享盘上的 12.8，不用镜像自带的 CUDA。`CUDACXX` / `CUDAToolkit_ROOT` 给编译和 CMake。

### 4. NCCL

```bash
export NCCL_HOME=/opt/huawei/dataset/Common_wl/pkgs_llb/nccl
export CPATH=${NCCL_HOME}/include:$CPATH
export LIBRARY_PATH=${NCCL_HOME}/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=${NCCL_HOME}/lib:$LD_LIBRARY_PATH
```

多机通信用共享盘这套 NCCL，避免和 conda / 系统 NCCL 混用。

### 5. 多机通信

```bash
export DEEPSPEED_TIMEOUT=3000
export NCCL_CONNECT_TIMEOUT=18000
export NCCL_NET_GDR_LEVEL=0
export NCCL_DEBUG=INFO
# export NCCL_P2P_DISABLE=1   # 只有 P2P 出问题再开
```

| 变量 | 作用 |
|---|---|
| `DEEPSPEED_TIMEOUT` | DeepSpeed 等集合通信超时（秒） |
| `NCCL_CONNECT_TIMEOUT` | NCCL 建链超时，多机要加大 |
| `NCCL_NET_GDR_LEVEL=0` | 关掉 GPU Direct RDMA，本集群更稳 |
| `NCCL_DEBUG=INFO` | 拉起阶段看 NCCL；稳定后可去掉 |
| `NCCL_P2P_DISABLE` | 默认注释；P2P 挂了再开 |

### 6. Conda：激活本机正在用的环境，再钉 CUDA / cuDNN

本机先 `conda activate` 调通，然后 `echo $CONDA_PREFIX`，把**同一条绝对路径**写进多卡脚本。不要写死某个公共环境名，也不要只写 `qwen36` 这种短名。

```bash
export MINICONDA_PATH=/opt/huawei/dataset/Common_wl/miniconda3
export PATH=$MINICONDA_PATH/bin:$PATH
$MINICONDA_PATH/bin/conda init bash
source "$MINICONDA_PATH/etc/profile.d/conda.sh"

# 写成和本机 `echo $CONDA_PREFIX` 完全一样的绝对路径
conda activate /本机调试用的环境绝对路径

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PATH=${CUDA_HOME}/bin:$PATH
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH

# cuDNN 跟刚激活的本机环境走（有 pip nvidia-cudnn 时）
CUDNN_HOME=$(ls -d "$CONDA_PREFIX"/lib/python*/site-packages/nvidia/cudnn 2>/dev/null | head -1 || true)
if [ -n "${CUDNN_HOME:-}" ]; then
    export CUDNN_HOME
    export CPATH=${CUDNN_HOME}/include:${CPATH:-}
    export LIBRARY_PATH=${CUDNN_HOME}/lib:${LIBRARY_PATH:-}
    export LD_LIBRARY_PATH=${CUDNN_HOME}/lib:$LD_LIBRARY_PATH
fi
```

MTP 镜像里 conda 未初始化，必须 `init` + `source conda.sh`。activate 用本机环境的绝对路径（须在共享盘上，两边都能读），不要只写短名。activate 会改 `PATH` / `LD_LIBRARY_PATH`，所以 CUDA 要再 export 一次。

### 7. vLLM 才需要

```bash
export VLLM_WORKER_MULTIPROCESS_METHOD=spawn
```

不用 vLLM 的任务可以不写。fork 在 3.12 + CUDA 上容易挂，vLLM worker 用 spawn。

## 多卡脚本骨架

```bash
#!/bin/bash
set -euo pipefail
# 1) 软链  2) SETUPTOOLS  3) CUDA/NCCL  4) NCCL 超时
# 5) conda activate 本机 $CONDA_PREFIX + 再导 CUDA/cuDNN
# 6) 业务：用 SCRIPT_DIR，不要写 /home/ma-user/work/algorithm/<算法包>/...

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# "$SCRIPT_DIR/train.py" ...
```

平台启动命令填该包在 MTP 下的脚本路径，例如 `bash /opt/huawei/schedule-train/algorithm/foo/run.sh`。
