# LingBot-VLA AlohaMini2Pro 适配工程

本工程用于把 **AlohaMini 2 Pro**（代码名 `alohamini2pro`）接入 LingBot-VLA。

当前是三台设备：

- 树莓派：机器人内网，只运行 `lerobot_alohamini` host。
- 本地 PC：和树莓派同一内网，运行真实机器人推理桥 `infer_robot.py`。
- GPU 服务器：公网机器，运行数据检查、norm stats、训练和 LingBot websocket 模型服务。

因为 GPU 服务器无法访问内网树莓派，所以真机推理桥不能放在服务器上。正确链路是：

```text
树莓派 host -> 本地 PC infer_robot.py -> GPU 服务器 LingBot websocket -> 本地 PC infer_robot.py -> 树莓派 host -> 机器人执行 action
```

## 结论

- 训练在 GPU 服务器。
- LingBot 模型推理在 GPU 服务器。
- 真实机器人推理桥在本地 PC。
- 树莓派启动和数据录制仍使用 `lerobot_alohamini`，不在本工程里实现。
- 后续运行本工程脚本时，推荐直接用脚本绝对路径，不要求手动设置 `PYTHONPATH`、`LINGBOT_VLA_REPO` 这类环境变量。

## 项目结构

```text
/home/yigeoooo/project/lingbotvla/
  configs/robot_configs/alohamini2pro.yaml       # LeRobot AlohaMini2Pro -> LingBot 特征映射
  configs/vla/alohamini2pro_real_load20000h.yaml # LingBot 训练配置
  scripts/verify_dataset.py                      # 检查 LeRobot 数据集是否是 Pro2 18 维
  scripts/compute_norm.py                        # 调 LingBot 官方 compute_norm.py
  scripts/train_lingbot.py                       # 调 LingBot 官方 train_lingbotvla.py
  scripts/start_lingbot_server.py                # 启动 LingBot websocket policy server
  scripts/infer_robot.py                         # 本地 PC 真实机器人推理桥
  src/lingbotvla_alohamini/                      # 适配代码
  third_party/lingbot-vla/                       # 官方 Robbyant/lingbot-vla 仓库
```

## 三端职责

### 树莓派

树莓派只做机器人硬件 host：

- 连接电机、相机、底盘、升降轴。
- 开 ZMQ host，给本地 PC 读取 observation 和发送 action。
- 不跑 LingBot 模型。
- 不安装本工程。

启动命令仍在 `lerobot_alohamini` 里执行：

```bash
python -m lerobot.robots.alohamini.lekiwi_host --robot_model alohamini2pro
```

参数含义：

- `--robot_model alohamini2pro`：指定 AlohaMini2Pro 18 维机器人模型。不要省略，否则可能按旧的 16 维模型处理。

### 本地 PC

本地 PC 是真机推理桥：

- 和树莓派在同一个内网。
- 能访问树莓派 `5555/5556`。
- 能访问公网 GPU 服务器，或者通过 SSH/VPN 隧道访问 GPU 服务器。
- 运行 `scripts/infer_robot.py`。
- 不加载 Qwen2.5-VL，不加载 LingBot checkpoint，不需要 GPU。

本地 PC 做的事情：

1. 从树莓派读取 camera/state。
2. 组织 LingBot observation payload。
3. 发给 GPU 服务器 LingBot websocket。
4. 收到 18 维 action。
5. 做速度、升降限幅。
6. 发回树莓派执行。

### GPU 服务器

GPU 服务器负责模型侧：

- 读取/检查 LeRobot v3 数据集。
- 计算 LingBot norm stats。
- 训练 LingBot-VLA。
- 启动 LingBot websocket policy server。
- 不访问树莓派内网。
- 不运行 `infer_robot.py`，除非以后你用 VPN/专线让服务器能直接访问树莓派。

## 环境依赖结论

不建议把 `lerobot_alohamini` 机器人环境和 LingBot 训练/模型环境共用一个 conda。

原因是依赖版本存在硬冲突：

| 依赖 | `lerobot_alohamini` 当前要求 | LingBot 当前要求 |
| --- | --- | --- |
| Python | `>=3.12` | `>=3.8`，建议也用 3.12 |
| torch | `>=2.7,<2.12` | `torch==2.8.0` |
| torchvision | `>=0.22,<0.27` | `torchvision==0.23.0` |
| numpy | `>=2.0,<2.3` | `numpy==1.26.4` |
| datasets | `>=4.7,<5` | `datasets==3.6.0` |
| transformers | LeRobot policy extra 为 `>=5.4,<5.6` | `transformers==4.51.3` |
| torchcodec | `>=0.3,<0.12` | `torchcodec==0.6.0` |

推荐环境划分：

- 树莓派：使用 `lerobot_alohamini` 的机器人环境。
- 本地 PC：使用 `lerobot_alohamini` 的机器人环境，再安装本工程这个轻量适配包。
- GPU 服务器：新建 LingBot 专用环境，按 LingBot 官方 `install.sh` 安装；不要在这个环境里安装 `lerobot_alohamini[all]`。

注意：LingBot 官方 `install.sh` 会安装 Hugging Face LeRobot `v0.4.2`，这是 LingBot 代码读取 LeRobot v3 数据集所需要的兼容版本。不要用本地 `lerobot_alohamini` 的完整依赖去覆盖它。

## 0. GPU 服务器环境

在 GPU 服务器新建 LingBot 专用环境：

```bash
conda create -y -n lingbotvla_server python=3.12
conda activate lingbotvla_server
```

安装 LingBot 官方依赖。这个步骤需要在 LingBot 仓库目录执行，因为官方 `install.sh` 使用了相对路径：

```bash
cd /home/yigeoooo/project/lingbotvla/third_party/lingbot-vla
bash install.sh
```

再安装本适配工程：

```bash
python -m pip install -e /home/yigeoooo/project/lingbotvla
```

快速检查：

```bash
python /home/yigeoooo/project/lingbotvla/scripts/verify_dataset.py --help
python /home/yigeoooo/project/lingbotvla/scripts/compute_norm.py --dry_run
python /home/yigeoooo/project/lingbotvla/scripts/train_lingbot.py --dry_run
```

## 1. 本地 PC 环境

本地 PC 只跑真实机器人推理桥。可以直接复用已有 `lerobot_alohamini` conda 环境；如果没有，就单独建一个 LeRobot 机器人环境。

```bash
conda create -y -n alohamini_bridge python=3.12
conda activate alohamini_bridge
python -m pip install -e "/home/yigeoooo/project/lerobot_alohamini[all]"
python -m pip install -e /home/yigeoooo/project/lingbotvla
```

检查：

```bash
python /home/yigeoooo/project/lingbotvla/scripts/infer_robot.py --help
```

本地 PC 不需要执行 LingBot 的 `install.sh`。

## 2. 数据录制

数据录制继续使用 `lerobot_alohamini` 原来的录制流程。本工程不写录制脚本。

关键要求：录制端 client 必须使用 `robot_model=alohamini2pro`，并且必须写入自然语言任务名。对 `lerobot_alohamini/examples/alohamini/record_bi.py` 这类脚本，就是传 `--task_description`。这个 task 会进入 LeRobot 数据集的 `meta/tasks.parquet` 和每帧 `task_index`，LingBot 训练时会把它转成语言 prompt。

合格数据应满足：

- `action.shape == (18,)`
- `observation.state.shape == (18,)`
- action/state 名称顺序一致
- 至少包含 `observation.images.chest`、`observation.images.wrist_left`、`observation.images.wrist_right`

如果录制出来是 16 维，大概率是用了旧的 `alohamini1` 配置，不能直接训练 Pro2。

task 名称不要用 `robot task`、`My task description4` 这类占位文本。建议直接写推理时也会使用的英文指令，例如 `pick up the red cube and place it in the box`。如果一个数据集混合多个任务，每个 episode 应有对应任务文本。

## 3. GPU 服务器检查数据格式

把 LeRobot v3 数据集放到 GPU 服务器可访问的位置，或者使用 Hugging Face repo id。

```bash
python /home/yigeoooo/project/lingbotvla/scripts/verify_dataset.py \
  --repo_id <HF_USER>/<DATASET_REPO>
```

如果是本地数据集，可以加：

```bash
python /home/yigeoooo/project/lingbotvla/scripts/verify_dataset.py \
  --repo_id <LOCAL_REPO_ID> \
  --root /path/to/local/lerobot/root
```

参数含义：

- `--repo_id`：LeRobot 数据集 repo id，可以是 Hugging Face repo，也可以是本地 LeRobot dataset id。
- `--root`：本地数据集根目录。只在数据集不从默认缓存/HF Hub 读取时需要。

检查输出会打印 `total_tasks` 和 `tasks` 列表。VLA 训练必须有 task；如果没有 task，或者 task 是明显占位文本，需要回到录制数据或数据集元数据里修正。

## 4. GPU 服务器计算 norm stats

先准备输出目录：

```bash
mkdir -p /home/yigeoooo/project/lingbotvla/assets/norm_stats
```

运行：

```bash
python /home/yigeoooo/project/lingbotvla/scripts/compute_norm.py \
  --data.data_name alohamini2pro \
  --data.train_path /path/to/lerobot_v3_dataset_or_hf_repo \
  --data.robot_config_root /home/yigeoooo/project/lingbotvla/configs/robot_configs \
  --data.norm_stats_file /home/yigeoooo/project/lingbotvla/assets/norm_stats/alohamini2pro.json
```

参数含义：

- `scripts/compute_norm.py`：本工程包装脚本，内部调用 LingBot 官方 `third_party/lingbot-vla/scripts/compute_norm.py`。
- `--`：分隔符。前面是本工程包装脚本参数，后面原样传给 LingBot 官方脚本。
- `--data.data_name alohamini2pro`：LingBot robot config 名称，必须对应 `configs/robot_configs/alohamini2pro.yaml` 文件名。
- `--data.train_path`：LeRobot v3 数据集路径或 Hugging Face repo id。
- `--data.robot_config_root`：robot config 目录。
- `--data.norm_stats_file`：norm stats 输出 JSON，训练和推理应使用同一份。

## 5. GPU 服务器训练 LingBot-VLA

先 dry-run 看最终调用的官方命令：

```bash
python /home/yigeoooo/project/lingbotvla/scripts/train_lingbot.py --dry_run
```

`train_lingbot.py` 包装脚本本身参数很少，真正的训练参数都放在 `--` 后面，原样传给 LingBot 官方 `tasks/vla/train_lingbotvla.py`。

当前 `configs/vla/alohamini2pro_real_load20000h.yaml` 已经给了一组默认训练参数：

- `train.max_steps=40000`
- `train.save_steps=10000`
- `train.micro_batch_size=8`
- `train.gradient_accumulation_steps=1`
- `train.lr=5.0e-5`
- `train.data_parallel_mode=fsdp2`
- `train.enable_resume=true`

建议正式训练时把关键参数显式写在命令里，方便复现实验：

```bash
python /home/yigeoooo/project/lingbotvla/scripts/train_lingbot.py \
  --data.data_name alohamini2pro \
  --data.train_path /path/to/lerobot_v3_dataset_or_hf_repo \
  --data.robot_config_root /home/yigeoooo/project/lingbotvla/configs/robot_configs \
  --data.norm_stats_file /home/yigeoooo/project/lingbotvla/assets/norm_stats/alohamini2pro.json \
  --train.output_dir /home/yigeoooo/project/lingbotvla/output/alohamini2pro \
  --train.max_steps 40000 \
  --train.save_steps 10000 \
  --train.micro_batch_size 8 \
  --train.gradient_accumulation_steps 1 \
  --train.lr 5.0e-5 \
  --train.data_parallel_mode fsdp2 \
  --train.enable_resume true
```

参数含义：

- `--`：分隔符。前面是本工程包装脚本参数，后面是 LingBot 官方训练参数。
- `--data.data_name`：robot config 名称，必须对应 `alohamini2pro.yaml`。
- `--data.train_path`：LeRobot v3 数据集路径或 Hugging Face repo id。
- `--data.robot_config_root`：robot config 目录。
- `--data.norm_stats_file`：第 4 步计算出的 norm stats。
- `--train.output_dir`：checkpoint、日志、配置保存目录。
- `--train.max_steps`：最大 optimizer update step 数。训练到这个 step 会停止，并保存最终 checkpoint。
- `--train.save_steps`：每隔多少 step 保存一次 checkpoint。
- `--train.micro_batch_size`：每张 GPU 每次 forward 的样本数，显存不够优先调小。
- `--train.gradient_accumulation_steps`：梯度累积步数。全局 batch 约等于 `micro_batch_size * GPU 数量 * gradient_accumulation_steps`。
- `--train.lr`：学习率。
- `--train.data_parallel_mode`：并行方式，常见值是 `ddp`、`fsdp1`、`fsdp2`。当前配置默认 `fsdp2`。
- `--train.enable_resume`：是否从 `output_dir` 下最新 checkpoint 自动恢复。

训练命令不单独传 `--task`。LingBot 会从 LeRobot 数据集里读取每帧的 `task_index`，再从 `meta/tasks.parquet` 取自然语言 task，内部构造 `prompt: [task]` 作为 VLA 语言输入。

显存不够时可以这样起步：

```bash
python /home/yigeoooo/project/lingbotvla/scripts/train_lingbot.py \
  --data.data_name alohamini2pro \
  --data.train_path /path/to/lerobot_v3_dataset_or_hf_repo \
  --data.robot_config_root /home/yigeoooo/project/lingbotvla/configs/robot_configs \
  --data.norm_stats_file /home/yigeoooo/project/lingbotvla/assets/norm_stats/alohamini2pro.json \
  --train.output_dir /home/yigeoooo/project/lingbotvla/output/alohamini2pro \
  --train.max_steps 20000 \
  --train.save_steps 5000 \
  --train.micro_batch_size 1 \
  --train.gradient_accumulation_steps 8 \
  --train.lr 5.0e-5 \
  --train.use_compile false
```

更多常用训练覆盖参数：

- `--train.num_train_epochs`：按 epoch 控制训练时长。通常和 `max_steps` 二选一，当前推荐用 `max_steps`。
- `--train.num_workers` 不存在；数据加载线程是 `--data.num_workers`。
- `--data.num_workers`：DataLoader workers，默认配置为 `8`。
- `--train.chunk_size`：action chunk 长度，默认 `50`，norm stats 会按这个 chunk 组织动作统计。
- `--train.tokenizer_max_length`：语言 token 最大长度，当前配置为 `72`。
- `--train.max_action_dim`、`--train.max_state_dim`：padding 后动作/状态最大维度，当前配置为 `75`。
- `--train.ckpt_manager`：checkpoint 管理方式，当前配置为 `dcp`。
- `--train.use_wandb`、`--train.wandb_project`、`--train.wandb_name`：可选 wandb 记录。

## 6. GPU 服务器启动 LingBot policy server

训练完成后，在 GPU 服务器启动模型服务：

```bash
python /home/yigeoooo/project/lingbotvla/scripts/start_lingbot_server.py \
  --ckpt /path/to/posttraining_ckpt \
  --qwen25_path /path/to/Qwen2.5-VL-3B-Instruct \
  --port 8006 \
  --use_length 1
```

参数含义：

- `--ckpt`：post-training checkpoint 目录，目录下必须有 `lingbotvla_cli.yaml`。
- `--qwen25_path`：Qwen2.5-VL 权重路径。脚本会在子进程里设置 `QWEN25_PATH`，不需要你手动 export。
- `--port`：websocket 服务端口。本地 PC 要连这个端口。
- `--use_length`：每次 websocket 返回的 action chunk 使用长度。建议真机先用 `1`。
- `--num_denoising_step`：采样步数，默认 `10`。显存或延迟压力大时可以调低。
- `--norm_path`：可选，手动指定 norm stats。通常 checkpoint 配置里已有，不需要传。
- `--use_compile`：可选，启用 `torch.compile`，首次启动会更慢，对环境要求更高。

安全要求：LingBot 官方 websocket server 默认监听 `0.0.0.0`，没有鉴权。不要把 `8006` 对全公网开放。推荐用 SSH 隧道，或者云安全组只允许本地 PC 的出口公网 IP 访问 `8006`。

## 7. 本地 PC 建立到 GPU 服务器的连接

推荐 SSH 本地端口转发。在本地 PC 开一个终端：

```bash
ssh -N -L 8006:127.0.0.1:8006 <USER>@<GPU_SERVER_PUBLIC_IP_OR_DOMAIN>
```

参数含义：

- `-N`：只建立 SSH 连接，不执行远程 shell 命令。
- `-L 8006:127.0.0.1:8006`：把本地 PC 的 `127.0.0.1:8006` 转发到 GPU 服务器的 `127.0.0.1:8006`。
- `<USER>@<GPU_SERVER_PUBLIC_IP_OR_DOMAIN>`：GPU 服务器 SSH 登录用户和公网 IP/域名。

如果不用 SSH 隧道，则本地 PC 需要能直接访问 GPU 服务器 `8006`。

## 8. 本地 PC 运行真实机器人推理桥

使用 SSH 隧道时：

```bash
python /home/yigeoooo/project/lingbotvla/scripts/infer_robot.py \
  --remote_ip <PI_IP> \
  --policy_host 127.0.0.1 \
  --policy_port 8006 \
  --task "pick up the object" \
  --fps 20 \
  --duration 45
```

不用 SSH 隧道、直接访问公网服务器时：

```bash
python /home/yigeoooo/project/lingbotvla/scripts/infer_robot.py \
  --remote_ip <PI_IP> \
  --policy_host <GPU_SERVER_PUBLIC_IP_OR_DOMAIN> \
  --policy_port 8006 \
  --task "pick up the object" \
  --fps 20 \
  --duration 45
```

参数含义：

- `--remote_ip`：树莓派内网 IP，本地 PC 必须能访问。
- `--policy_host`：LingBot websocket 地址。SSH 隧道时填 `127.0.0.1`；公网直连时填 GPU 服务器 IP/域名。
- `--policy_port`：LingBot websocket 端口，默认 `8006`。
- `--task`：自然语言任务指令，会作为 LingBot 输入。应尽量和训练数据里的 task 文本一致，至少语义一致。
- `--fps`：本地 PC 控制循环频率。
- `--duration`：本次推理持续秒数。
- `--max_xy_vel`：底盘 x/y 最大速度限幅，默认 `0.25`。
- `--max_theta_vel`：底盘旋转速度限幅，默认 `75.0`。
- `--min_lift_mm`、`--max_lift_mm`：升降轴高度限幅，默认 `0` 到 `600`。

真机第一次测试建议把 `--duration` 设短一点，例如 `5` 到 `10` 秒，并确认急停方案。

## 9. 可选：服务器直接跑推理桥的条件

只有当 GPU 服务器能直接访问树莓派 `5555/5556` 时，才可以在 GPU 服务器运行 `infer_robot.py`。你当前拓扑是树莓派内网、服务器公网，默认不要这样做。

## 当前限制和注意事项

- 本工程不启动树莓派端，不录制数据。
- 当前默认只用 3 路 RGB 相机：`chest`、`wrist_left`、`wrist_right`。
- 当前未启用 depth。
- 不要直接用当前 `lerobot-rollout` 跑全身策略；它会过滤非 `.pos` 动作，导致 `x.vel/y.vel/theta.vel/lift_axis.height_mm` 被丢掉。
- 真实机器人推理前先短时长、低速度测试。
