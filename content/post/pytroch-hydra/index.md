---
title: "PyTorch Lightning + Hydra 工程化实践"
description: 
date: 2025-11-23T21:43:40+08:00
image: 
hidden: false
tags:
  - pytorch
  - hydra
  - lightning
categories:
  - code
---

这个架构利用 **PyTorch Lightning** 处理训练循环，利用 **Hydra** 处理复杂的参数配置和实验管理。

## 前置准备

安装必要的库：

Bash

```bash
pip install pytorch-lightning hydra-core tensorboard
```

## 项目目录结构

Hydra 严格依赖文件结构，请确保你的项目符合以下层级：

Plaintext

```jsx
my_project/
├── conf/                   # [配置文件夹]
│   ├── config.yaml         # 主入口配置
│   ├── model/              # 模型参数组
│   │   └── default_model.yaml
│   └── train/              # 训练参数组
│       └── default_train.yaml
├── src/                    # [源码文件夹]
│   ├── __init__.py
│   └── model.py            # LightningModule 定义
└── main.py                 # [主入口] 连接 Hydra 和 Lightning
```

---

## 配置文件编写 (YAML)

### `conf/model/default_model.yaml`

定义模型的超参数。

YAML

```yaml
# 模型相关参数
input_dim: 28
hidden_dim: 64
learning_rate: 0.01
```

### `conf/train/default_train.yaml`

定义 Trainer 的参数。

YAML

```yaml
# 训练器相关参数
max_epochs: 10
accelerator: "auto"
devices: 1
batch_size: 32
```

### `conf/config.yaml`

主配置文件，用于组合各个模块。

YAML

```yaml
defaults:
  - model: default_model    # 默认使用 conf/model/default_model.yaml
  - train: default_train    # 默认使用 conf/train/default_train.yaml
  - _self_                  # 固定写法，表示覆盖顺序

# 全局通用配置
seed: 42
project_name: "hydra_lightning_demo"
```

---

## 模型代码 (Python)

### `src/model.py`

保持模型代码纯净，只在 `__init__` 接收参数。

Python

```python
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

class LitModel(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, learning_rate):
        super().__init__()
        # 关键：保存超参数，这会将参数写入 hparams.yaml 并记录到日志中
        self.save_hyperparameters()
        
        self.layer1 = nn.Linear(input_dim * input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.layer2(F.relu(self.layer1(x)))

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        
        # 记录日志
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        # 使用 self.hparams 访问保存的参数
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
```

---

## 主程序 (Python)

### `main.py`

这是整个系统的控制中心。

Python

```python
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.model import LitModel

# 1. 设置 Hydra 装饰器，指向 conf 目录
@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print(f"--- Current Configuration ---\n{OmegaConf.to_yaml(cfg)}")
    
    # 设置随机种子
    pl.seed_everything(cfg.seed)

    # 2. 实例化模型
    # 从 cfg.model 中读取参数
    model = LitModel(
        input_dim=cfg.model.input_dim,
        hidden_dim=cfg.model.hidden_dim,
        learning_rate=cfg.model.learning_rate
    )

    # 模拟数据 (通常这里会使用 LightningDataModule)
    fake_data = torch.randn(100, 1, 28, 28)
    fake_labels = torch.randint(0, 10, (100,))
    train_loader = DataLoader(TensorDataset(fake_data, fake_labels), batch_size=cfg.train.batch_size)

    # 3. 配置 Logger
    # Hydra 会自动切换工作目录到 outputs/日期/时间，
    # 所以 save_dir 设置为 "." 表示当前 Hydra 生成的目录
    logger = TensorBoardLogger(save_dir=".", name="", version="")

    # 4. 实例化 Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.train.max_epochs,
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
        logger=logger,
        enable_progress_bar=True
    )

    # 5. 开始训练
    trainer.fit(model, train_loader)

if __name__ == "__main__":
    main()
```

---

## 如何运行实验

在终端中执行以下命令，体验 Hydra 的强大功能。

### 基础运行

使用默认配置运行：

```Bash
python main.py
```

*注：日志会自动保存在 `outputs/YYYY-MM-DD/HH-MM-SS/` 目录下。*

### 动态修改参数 (Override)

不修改代码，直接在命令行改变学习率和 Epoch：


```bash
python main.py model.learning_rate=0.001 train.max_epochs=20
```

### 多重运行 (Multirun / Grid Search)

**这是对比实验的核心功能**。一条命令运行 4 组实验（2种 hidden_dim * 2种 learning_rate）：

Bash

```bash
python main.py -m model.hidden_dim=64,128 model.learning_rate=0.01,0.001
```

*注：Hydra 会在 `multirun/` 目录下为每组实验生成独立的子文件夹和日志。*

### 进阶技巧：使用 `instantiate` (更优雅的写法)

上面的 `main.py` 中，我们手动把 `cfg.model.xxx` 传给了 `LitModel`。如果你参数很多，这很麻烦。
Hydra 提供了一个高级功能：**Object Instantiation**。

1. **修改 YAML**，加入 `_target_` 指向类路径：YAML
    
    ```yaml
    # conf/model/small.yaml
    _target_: src.model.LitModel  # <--- 关键
    input_dim: 28
    hidden_dim: 64
    learning_rate: 0.01
    ```
    

---

1. **修改 `main.py`**，使用 `hydra.utils.instantiate`：
    
    ```python
    import hydra
    
    @hydra.main(...)
    def main(cfg: DictConfig):
        # 这行代码会自动寻找 _target_ 指向的类，
        # 并把 yaml 里剩下的参数全部传给这个类的 __init__
        model = hydra.utils.instantiate(cfg.model) 
    
        trainer = pl.Trainer(...)
        trainer.fit(model, ...)
    ```
    

## 查看结果

使用 TensorBoard 查看所有实验的对比：

Bash

```bash
# 如果是单次运行
tensorboard --logdir outputs

# 如果使用了 -m 多重运行
tensorboard --logdir multirun
```

在 TensorBoard 的 **HPARAMS** 标签页中，你可以直观地看到 Parallel Coordinates 图，展示 `learning_rate` 和 `hidden_dim` 如何影响最终的 Loss。

## 修改log位置

修改保存地址和文件名主要涉及到两个层面：

1. **文件夹结构（Hydra 控制）**：决定了整个实验的输出目录在哪里（例如从默认的 `outputs/日期/时间` 改为 `experiments/实验名/`）。
2. **模型文件名（PyTorch Lightning 控制）**：决定了 `.ckpt` 权重文件叫什么名字（例如从 `epoch=0-step=100.ckpt` 改为 `best-acc=0.98.ckpt`）。

以下是具体的修改方法：

---

### 1. 修改输出文件夹路径 (Hydra 配置)

默认情况下，Hydra 会在 `outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}` 创建目录。要自定义它，你需要在 **`conf/config.yaml`** 中添加 `hydra` 配置节点。

**修改 `conf/config.yaml`：**

YAML

```yaml
defaults:
  - model: default_model
  - train: default_train
  - _self_

# 定义一个项目名或实验名
experiment_name: "my_custom_experiment"

# --- Hydra 配置核心 ---
hydra:
  run:
    # 修改单次运行的输出路径
    # 示例：./logs/my_custom_experiment/2023-10-27_10-30/
    dir: ./logs/${experiment_name}/${now:%Y-%m-%d_%H-%M-%S}
  
  sweep:
    # 修改多重运行 (-m) 的根目录和子目录
    dir: ./multirun_logs/${experiment_name}/${now:%Y-%m-%d_%H-%M-%S}
    subdir: ${hydra.job.num} # 子文件夹名为 0, 1, 2...
```

这样设置后，你的实验文件就不会散落在 `outputs` 里，而是整齐地存放在 `logs/` 下。

---

### 2. 修改日志文件夹名称 (Logger 配置)

在 `main.py` 中初始化 `TensorBoardLogger` 时，如果你的 Hydra 路径已经很精确了（比如上面已经精确到了秒），通常我们希望 Logger **不要再创建子文件夹**（不要再搞个 `version_0`），而是直接写在当前目录。

**修改 `main.py`：**

Python

```python
# "." 表示直接使用 Hydra 创建好的那个目录，不创建新的子文件夹
# name="" 和 version="" 能够去除额外的目录层级
logger = TensorBoardLogger(save_dir=".", name="", version="")
```

---

### 3. 修改模型权重文件名 (Checkpoint 回调)

这是最重要的一步。你需要使用 PyTorch Lightning 的 `ModelCheckpoint` 回调来自定义保存的文件名（例如包含验证集准确率）。

**修改 `main.py`：**

Python

```python
from pytorch_lightning.callbacks import ModelCheckpoint

# ... 在 main 函数中 ...

def main(cfg: DictConfig):
    # ... 模型初始化代码 ...

    # --- 定义 Checkpoint 回调 ---
    checkpoint_callback = ModelCheckpoint(
        dirpath=".",                  # 保存路径：直接存在当前 Hydra 目录下
        filename="best-{epoch:02d}-{val_loss:.2f}", # 文件名格式
        monitor="val_loss",           # 监控的指标 (需在 training_step 用 self.log 记录过)
        mode="min",                   # "min" 表示 loss 越小越好
        save_top_k=1,                 # 只保存最好的 1 个模型
        save_last=True                # 额外保存一个 last.ckpt (最新的模型，以防中断)
    )

    trainer = pl.Trainer(
        # ... 其他参数 ...
        callbacks=[checkpoint_callback], # 注册回调
        logger=logger
    )
    
    trainer.fit(model, train_loader)
```

> 注意：你在 filename 里使用的 {val_loss} 必须在你的 LitModel 代码里通过 self.log("val_loss", ...) 记录过，否则会报错。
> 

---

### 4. 完整整合示例

把你现在的 `conf/config.yaml` 和 `main.py` 更新为以下内容：

### `conf/config.yaml`

YAML

```yaml
defaults:
  - model: default_model
  - train: default_train
  - _self_

experiment_name: "resnet_cifar10"

hydra:
  run:
    # 最终路径: logs/resnet_cifar10/2023-xx-xx_12-00-00/
    dir: logs/${experiment_name}/${now:%Y-%m-%d_%H-%M-%S}
  sweep:
    dir: multirun/${experiment_name}/${now:%Y-%m-%d_%H-%M-%S}
    subdir: run_${hydra.job.num}
```

### `main.py`

Python

```python
import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
# ... import your model ...

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    pl.seed_everything(42)
    
    model = ... # 初始化你的模型

    # 1. 设置日志器：直接写在 Hydra 目录下
    logger = TensorBoardLogger(save_dir=".", name="", version="")

    # 2. 设置模型保存规则
    # 假设模型里 log 了 'val_acc'
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",  # 会在 Hydra 目录下创建一个 checkpoints 文件夹
        filename="{epoch:02d}-{val_acc:.4f}", # 比如: 05-0.9821.ckpt
        monitor="val_acc",
        mode="max",
        save_top_k=1
    )

    trainer = pl.Trainer(
        max_epochs=cfg.train.max_epochs,
        logger=logger,
        callbacks=[checkpoint_callback],
        # ...
    )

    trainer.fit(model, ...)

if __name__ == "__main__":
    main()
```

### 最终效果

运行后，你的文件结构会非常清晰：

Plaintext

logs/
└── resnet_cifar10/                 <-- config.yaml 中的 experiment_name
    └── 2023-11-20_10-30-00/        <-- 时间戳
        ├── .hydra/                 <-- Hydra 的配置备份
        ├── checkpoints/            <-- ModelCheckpoint 创建的文件夹
        │   └── 05-0.9821.ckpt      <-- 自定义的文件名
        ├── events.out.tfevents...  <-- TensorBoard 日志
        └── hparams.yaml            <-- PL 自动保存的参数