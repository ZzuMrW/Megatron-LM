# Megatron Core 0.18.0 接入 MindIO TFT 26.2.0

在已有的 Ascend Megatron 训练任务上接入 TFT，可以保留原生 DDP、分布式优化器和数据采样器。Megatron 的多实例分布式优化器（Partial DistOpt）提供同一参数分片的多个副本。发生故障时，TFT 指定恢复步、提供数据的 rank 和负责写盘的 rank。适配代码要把这些决定对应到 Megatron 实际持有的参数、优化器状态和训练进度。

本指南覆盖临终 checkpoint、HBM 不可纠正错误（UCE）的步级恢复，以及在原作业中替换进程的 ARF。代码移植自官方 MindSpeed-LLM `26.2.0@effa8e78` 默认源码树。该版本设置 `MINDSPEED_LLM_VERSION=018` 时会切换 Python 包的加载路径；表中复用的 TFT 回调位于默认树。目标版本是 Megatron-LM `core_v0.18.0@ba7b5ebc`，SDK 协议取自 MindIO `v26.2.0@6cb6683c`。这些版本标明了代码来源。多节点恢复、临终文件读回及性能仍待客户环境确认。

## 一、在客户训练分支接入补丁

在干净的目标版本上可直接应用[适配补丁](megatron-core-v0.18.0-mindspeed-llm-26.2-tft.patch)。从客户 Megatron 仓库根目录执行下列命令，把变量设为补丁的实际位置。已有 Ascend 适配的分支若在相同位置修改过代码，按表中的调用时序合并，保留客户已有的设备绑定和训练逻辑。

```bash
TFT_PATCH=/path/to/megatron-core-v0.18.0-mindspeed-llm-26.2-tft.patch
git apply --check "$TFT_PATCH"
git apply "$TFT_PATCH"
```

除新增的 [high_availability 目录](../../megatron/training/high_availability)外，补丁修改三处已有文件：

| 客户仓接入位置 | 修改及原因 |
| --- | --- |
| [训练初始化](../../megatron/training/initialize.py)中的 `_initialize_distributed` | 在创建默认通信组（world）前按 `LOCAL_RANK` 绑定 NPU 并启动 TFT。processor 在此时识别替换进程，Megatron 才能决定哪些启动期通信要推迟。如果 launcher 已经创建了 world，应先调整这两个动作的顺序；当前入口遇到已初始化的 world 会报错。 |
| [训练主循环](../../megatron/training/training.py)中的 `pretrain`、`train_step`、`train` 和数据加载入口 | 模型、优化器和 scheduler 就绪后，注册回调并用 TFT 装饰器调用原生 `train`。在顶层 `optimizer.step()` 之前上报开始更新，在设备同步、样本计数、scheduler 和该步进度就绪后上报完成并响应暂停。显式跳过的迭代也会推进数据游标，需要单独记录。回退后，训练使用重建的 iterator。 |
| [分布式优化器](../../megatron/core/optimizer/distrib_optimizer.py)中的 `load_parameter_state_from_dp_zero` | legacy 加载分两次完成。optimizer 元数据先恢复 Adam 的标量 `step`，随后 DP-zero scatter 写回主权重和两个 moment。补丁在写回张量前保留已经恢复的 `step`，避免续训时权重与 Adam 更新次数不一致。这个改动只涉及 checkpoint 读取。 |

### 适配文件目录树

目录树以 Megatron 仓库根目录为起点，列出补丁涉及的 18 个代码文件和 `docs/ha` 下的交付文件。每行右侧标明文件的职责。

```text
Megatron-LM/
├── megatron/
│   ├── core/
│   │   └── optimizer/
│   │       └── distrib_optimizer.py             # legacy DP-zero 加载时保留已恢复的 Adam step
│   └── training/
│       ├── initialize.py                       # 在 world 建立前启动 TFT，识别 ARF 替换进程
│       ├── training.py                         # 接入训练装饰器、整步提交和回退后的数据迭代器
│       └── high_availability/
│           ├── __init__.py                     # 向训练入口提供 TFT 启用、启动和步更新接口
│           ├── high_availability_helper.py      # 解析 HIGH_AVAILABILITY 并设置各能力开关
│           ├── initialize_patch.py             # 在 world 建立前绑定 NPU 并启动 TFT
│           ├── tft_train_initialize.py          # 启动 controller/processor，注册回调，保存完成步进度
│           ├── tft_replica_group.py             # 绑定原生优化器域与副本，管理临时 HCCL/Gloo 修复组
│           ├── native_distopt.py                # 读取原生分片区间，校验并写回 Adam 张量和元数据
│           ├── tft_dump.py                      # 校验 TFT 保存分配，重组 legacy 文件并更新 tracker
│           ├── tft_stop_clean.py                # 停设备、判断 UCE 等级、清理通信句柄并恢复设备
│           ├── tft_tensor_recovery.py           # 遍历持久张量做安全标记，回退时恢复模型 buffer
│           ├── tft_optimizer_data_repair.py     # 分派修复动作，点对点传输状态、分片和日志，或加载磁盘
│           ├── tft_rollback.py                  # 恢复计数和 scheduler、同步模型参数、还原 RNG
│           ├── dataloader_patch.py              # 按已消费位置重建 iterator，隔离建迭代器时的 RNG
│           ├── communication_patch.py           # 记录原生组创建参数，处理替换进程的启动期通信
│           ├── tft_arf_group_repair.py          # 按 TFT 通知重连 HCCL、重建 Gloo 并重绑句柄
│           └── utils.py                        # 定义 clean 返回码和 UCE 等级等回调常量
└── docs/
    └── ha/
        ├── tft-26.2-adaptation-guide-zh.md                     # 客户接入步骤、改动原因和使用边界
        ├── tft-26.2-migration.md                              # 官方来源与目标函数的逐项对照
        └── megatron-core-v0.18.0-mindspeed-llm-26.2-tft.patch  # 应用到 core_v0.18.0 的实现补丁
```

## 二、准备运行环境与启动参数

先确认客户的 Ascend 训练栈已能运行原生 BF16 Adam 和 Partial DistOpt。示例以固定 world size、HCCL、可用的 Gloo 组和至少两个 optimizer 实例为前提，使用`single` 或 `cyclic` sampler 和 `--save`。以下是两实例任务应并入现有启动命令的参数片段；模型、数据和并行参数沿用客户配置。

```text
--bf16 --optimizer adam --use-distributed-optimizer
--num-distributed-optimizer-instances 2
--rerun-mode disabled --num-workers 0
--save <checkpoint目录>
```

原生优化器在每个实例内分片。Dense 使用 DP×CP 域，Expert 使用 EDP 域，各域大小须能被实例数整除。程序还会检查域内 rank 顺序、实例内分片位置以及跨实例组是否一致。通过整除检查只能说明分片大小可行；只有对应位置确实持有相同分片时，TFT 选出的 rank 才能提供所需数据。FSDP、precision-aware optimizer、CPU offload、FP8/FP4 和异步保存超出本示例的适配范围。

| 配置项 | 客户需要设置或确认的内容 |
| --- | --- |
| `HIGH_AVAILABILITY` | `dump` 开启基础 HA 和临终保存；`retry` 开启基础 HA 和 UCE；`recover` 开启基础 HA 和 ARF。需要三项能力时使用 `export HIGH_AVAILABILITY=retry,recover`。开关可用逗号组合，大小写不敏感。 |
| TFT wheel | 部署时按 `mindio_ttp` 发行版本 `26.2.0` 和 wheel 内的 `mindio_ttp/VERSION` 核对来源。对应的 Git 标识为 TFT 组件最后一次变更 `9a28823c002d113d109c369e4faf8790ddcd8d0a`；仓库发布 tag 指向 `6cb6683c`。应以组件标识核对 wheel。 |
| 地址、端口与 controller | `CONTROLLER_ADDR` 和 `PROCESSOR_ADDR` 沿用 26.2.0 的配置，默认地址回退到 `TTP_ADDR`，端口为 8000。设置 `MINDX_TASK_ID` 时由 MindX 提供外置 controller，否则由 rank 0 启动 controller。需要替换 rank 0 的部署应使用独立于该训练进程的 controller。当前补丁沿用来源代码的非 TLS 启动参数；现网要求 TLS 时，须先适配 SDK 启动参数与证书配置。 |
| ARF 运行环境 | 启动器应能在原作业中恢复相同的逻辑 rank 和 world size，并接入原作业 rendezvous。TorchNPU 需提供 `reinit_process_group` 及 HCCL 原句柄重连能力。替换进程还要取得原 rank 的启动元数据，具体接口见下文。 |

## 三、逐函数迁移表

表中 `core/HA` 和 `tasks/HA` 分别指官方 MindSpeed-LLM 默认树的 `mindspeed_llm/core/high_availability` 和 `mindspeed_llm/tasks/high_availability`。目标模块位于 Megatron 的 `megatron/training/high_availability`，训练入口和 legacy reader 的改动见第一节。合并客户分支时，可按“保留的行为”核对调用顺序，再按“为什么要改”定位依赖原优化器的部分。[逐函数迁移表与接入边界](tft-26.2-migration.md)

| 官方文件及函数 | 保留的行为 | 为什么要改，以及如何替换 | 目标位置 |
| --- | --- | --- | --- |
| `tasks/HA/high_availability_helper.py::get_env_args` | 解析 `dump`、`retry`、`recover` 及其组合。 | 客户用原生 trainer 启动，直接解析环境变量即可。首批范围为临终、UCE 和 ARF。 | [high_availability_helper.py](../../megatron/training/high_availability/high_availability_helper.py) |
| `tasks/HA/initialize_patch.py::initialize_distributed_wrapper` | 绑定设备、启动 TFT，再建立 world 和模型并行组。 | 原生 Partial DistOpt 已建副本组。移除 `ttp_initialize_replica_dp_group` 调用，在原生组就绪后读取成员。 | `_initialize_distributed`、[initialize_patch.py](../../megatron/training/high_availability/initialize_patch.py) |
| `core/HA/tft_train_initialize.py::tft_init_controller_processor` | 沿用地址回退、端口、MindX 部署和 processor 启动后的 reboot 判断。 | SDK 的框架名接口不接受 `Megatron`。适配使用通用 PyTorch 回调入口，显式设置 `TTP_STOP_CLEAN_BEFORE_DUMP=1`，保留故障保存前的 stop/clean。 | [tft_train_initialize.py](../../megatron/training/high_availability/tft_train_initialize.py) |
| 同文件 `tft_register_processor` | 先上报 optimizer 副本和普通 DP 组，再依次注册 save、rename、stop、clean、repair、rollback、rebuild、stream sync。 | 以原生实例数替代固定副本数。Dense 和 Expert 沿用 TFT 的域编号，各域绑定实际所属的子优化器（leaf）。 | `tft_train_initialize.py`、[tft_replica_group.py](../../megatron/training/high_availability/tft_replica_group.py) |
| 同文件 `tft_train/tft_train_wrapper` | 用 `tft_exception_handler` 包装训练。替换 rank 等待 TFT 通知并完成建组后报告 `ARF FINISH`。 | 原实现直接导入 LLM trainer，回调按固定下标取训练参数。0.18.0 将模型、优化器、scheduler 和 iterator 保存在显式训练上下文中，重入时取得当前 iterator。 | `training.py::pretrain`、`tft_train_initialize.py` |
| `core/HA/tft_replica_group.py::build_repair_group/get_repair_group/destroy_repair_group` | 按 TFT 的 `rank_list` 创建临时 HCCL/Gloo 修复组，回退后释放。 | 常驻的 `build_dp_cp_replica_group/build_dp_ep_replica_group` 改为读取原生 intra/inter 组。controller 会排序，上报前要核对排序后的相同位置仍对应同一 shard。 | `tft_replica_group.py` |
| `core/HA/tft_replica_optimizer*.py`、`tft_replica_chainedoptimizer.py` 的 step，以及官方 `training.py::train` | 保留 TFT 更新开始、完成、暂停的时序。 | 一个顶层 optimizer 可能含多个 leaf。逐个 leaf 上报会暴露只更新了一部分的步；适配把整次更新纳入同一事务，以实际训练进度替代旧 `current_step` 和 buffer cache。 | `training.py::train_step/train`、`TrainingState.begin/commit` |
| `core/HA/tft_dump.py::tft_save_callback/get_checkpoint_name/save_checkpoint/tft_rename_callback` | 使用 TFT 指定的参与者和名单首项 writer，先写临时目录，再按 rename 回调发布。 | `set_dump_args/need_write_file/sync_gather_all_model_params` 和旧保存方法依赖旧副本组。目标代码从原生 bucket 读取分片，按逻辑位置组装为 legacy 文件。 | [tft_dump.py](../../megatron/training/high_availability/tft_dump.py)、[native_distopt.py](../../megatron/training/high_availability/native_distopt.py) |
| `core/HA/tft_stop_clean.py::stop_callback/clean_callback/model_update_npu_tensor_to_safe/unset_gather_handle/torch_sync` | 停设备、判断 UCE 等级、清理 DDP 句柄，按 retry 恢复 HCCL，再重启设备。 | 原生 optimizer 没有旧 `optim.update_npu_tensor_to_safe()` 方法。应遍历实际 leaf、模型和 DDP buffer、Adam 状态所占的存储。 | [tft_stop_clean.py](../../megatron/training/high_availability/tft_stop_clean.py)、[tft_tensor_recovery.py](../../megatron/training/high_availability/tft_tensor_recovery.py) |
| `core/HA/tft_optimizer_data_repair.py::repair_callback/send_rank_repair/recv_rank_repair/save_and_send_ckpt/recv_ckpt_from_peer` | 按 TFT 的 `repair_type` 建组并分派收发；先传带长度的序列化元数据，再点对点传 optimizer 张量，最后传日志值。 | 旧 `state_dict_memory/state_dict_by_idx/load_state_dict_memory` 和 `send_optim_param_state/recv_and_load_optim_param_state` 均属于副本优化器。目标代码按原生 leaf 和参数区间核对状态键、形状及类型，并核对接收 rank 的完成步进度。 | [tft_optimizer_data_repair.py](../../megatron/training/high_availability/tft_optimizer_data_repair.py)、`native_distopt.py` |
| 同文件 `load_ckpt_repair` | 收到 TFT 指定的磁盘动作后加载周期 checkpoint，上报实际加载步。 | 0.18.0 的原生 loader 包含全 world 集合通信，必须全体参与。适配直接使用原生 loader，省去旧精度处理工具改写 tracker 的操作。 | `tft_optimizer_data_repair.py` |
| `core/HA/tft_rollback.py::rollback_callback/feature_rollback/build_dataset/rebuild_global_vars/gather_model_params_from_optimizer`、`tasks/HA/dataloader_patch.py` | 清理后先 copy 再 gather，重建 iterator，同步后释放修复组，恢复训练位置和日志状态。 | 按原生 leaf 调用 `_copy_main_params_to_model_params`，并用 DDP 的 `start_param_sync(force_sync=True)` 同步模型。加载 scheduler 前复位步数，按已消费样本重建原生 iterator。 | [tft_rollback.py](../../megatron/training/high_availability/tft_rollback.py)、[dataloader_patch.py](../../megatron/training/high_availability/dataloader_patch.py) |
| `tasks/HA/communication_patch.py::*wrapper`、`core/HA/tft_arf_group_repair.py::update_arf_reboot_flag/arf_rebuild_process_group_callback/init_*/update_model_and_optim_related_group` | 替换进程先在本地启动，保存原 load 路径；TFT 通知后重建 world、并行组和 optimizer 组，重绑句柄并恢复路径。 | 新进程可能从未创建旧 Gloo 句柄。按原生建组时记录的成员重建 Gloo，HCCL 用原句柄重连。原生 optimizer 启动依赖键交换的返回值，需要重放原始结果。 | `initialize_patch.py`、[communication_patch.py](../../megatron/training/high_availability/communication_patch.py)、[tft_arf_group_repair.py](../../megatron/training/high_availability/tft_arf_group_repair.py) |

## 四、故障发生时，怎样恢复到同一步

### 临终保存时怎样确定分片顺序和写盘 rank

假设两个实例分别包含 `[0,1,2,3]` 和 `[4,5,6,7]`。rank 0 和 4 持有相同位置的分片。若 TFT 选出 `[4,1,2,3]`，rank 4 提供逻辑分片 0，也是 TFT 指定的 writer。若临时 Gloo 组的实际顺序为 `[1,2,3,4]`，直接拼接收集结果就会把分片 0 写到末尾。因此，保存回调先确认每个逻辑位置恰有一份数据，再读取通信组的实际顺序，按原生 bucket 的区间重新排列。

writer 将模型和训练元数据写入 `model_optim_rng.pt`，将原生优化器分片写入 `distrib_optim.pt`。TFT 通知 rename 时，回调核对各 TP/PP 位置的文件存在且非空，发布 `iter_<step>` 目录后再替换 tracker。已含 checkpoint 的目录由 rename 的文件系统语义保护，tracker 指向相同或更晚的步时本次发布终止。

| 周期 checkpoint 配置 | 临终文件位置 | 重新启动时的加载参数 |
| --- | --- | --- |
| `--ckpt-format torch` | `<save>` 下的 legacy checkpoint | `--ckpt-format torch --load <save>` |
| `--ckpt-format torch_dist` | 独立的 `<save>.tft` 目录 | `--ckpt-format torch --load <save>.tft` |

当前临终回调要求 `EP=1`，同一模型文件的各域收到完全相同的有序保存名单。只要名单不同，即使 writer 相同，也会拒绝发布。legacy 文件采用原生的单份 RNG 记录语义。若必须恢复每个失败 rank 各自的 RNG，或要求故障产物仍为 `torch_dist`，还需确定额外的状态来源或保存规划；现有 TFT `save_info` 只提供参与者和域。

### UCE 修复分别从哪里取得优化器分片和训练进度

clean 处理设备和存储的可用性。高等级 UCE 可能已经损坏张量的值，设备重启后仍须按 TFT 指定的来源修复数据。接收方先确认源和目标属于同一域、持有相同逻辑分片，并核对参数区间和 Adam 状态描述。全部待恢复 leaf 的张量收齐后，才写回主权重、`exp_avg`、`exp_avg_sq` 和相应 step。

设 TFT 选择恢复到已完成步 S。对端可以提供该步的优化器分片，但不同 DP rank 消费的样本和使用的随机数可能不同。回退使用原 rank 在 S 的样本计数、RNG 和模型 buffer，复位 scheduler 步数再加载状态，从主权重同步模型参数，按实际消费位置重建 iterator，随后恢复 RNG。下一次完整训练为 S+1。原生训练已记录完成步的日志，回退记录收到的日志值，并重置 timer 和 rerun 状态。

若 TFT 下发 `RT_LOAD_CKPT` 或 `RT_LOAD_REBUILD`，适配器要求所有 rank 参与，且已配置 `--load`。加载后向 TFT 上报磁盘中的实际步数，并从该步回退。缺少可用内存进度时，磁盘恢复也须由 TFT 选择。

### ARF 替换进程怎样重新加入原作业

替换进程启动时，幸存进程正在等待 TFT 动作。此时直接执行普通启动的 barrier，或再次进行全 world 的参数组键交换，都可能等不到其他 rank。适配器会暂存 `load` 和 `pretrained_checkpoint`，推迟 Gloo 建组和普通数据迭代器启动。TFT 通知重建后，各参与 rank 按记录的成员恢复 HCCL/Gloo 通信，重绑引用。替换进程确认通信已重建，再通过 `ARF FINISH` 进入 TFT 后续动作。

使用 ARF 时，需要在客户的调用入口通过 `pretrain(..., tft_recovery=provider)` 接入已有状态源。两个方法解决的问题不同：

| 客户接口 | 返回内容与调用条件 | 缺失时的影响 |
| --- | --- | --- |
| `optimizer_group_keys(rank, exchange)` | 返回原生 optimizer 在第 `exchange` 次 world `all_gather_object` 时取得的逐 rank 参数组键，按原 world rank 排列，编号从 0 开始。替换进程校验结果长度及本 rank 的键。 | 原生工厂依赖这个返回值构造参数组。缺少结果或本地键不一致时，替换进程的模型/优化器初始化会失败。 |
| `rank_progress(rank, step)` | 当前进程没有指定步的小状态时，返回原逻辑 rank 在该完成步的 `rank`、`step`、`counters`、`flops`、`scheduler`、`rng`、`model_buffers`。其中 `counters` 包含 `consumed_train_samples`、`skipped_train_samples`、`consumed_valid_samples`。结构以 [TrainingState._progress](../../megatron/training/high_availability/tft_train_initialize.py) 为准。 | 同进程 UCE 可使用本地同一步的内存进度。原进程退出后，这份内存随之丢失；客户无法提供原 rank 状态时，内存修复无法继续，只能处理 TFT 正式下发的磁盘动作。 |

这两个方法由客户已有的数据来源提供。补丁本身只在进程内保存完成步进度。若客户自定义模型在启动时还有其他依赖返回值的集合通信，需要先确认替换 rank 能从哪里取得相同结果。外部数据流或带持久 worker 的 DataLoader 同样需要原游标和 worker 状态；当前回退路径使用原生普通 DataLoader。

## 五、客户接入时要确认的边界

| 客户配置或需求 | 当前实现与接入条件 |
| --- | --- |
| 临终保存时 `EP>1`，或同一 legacy 文件的各域收到不同名单 | 回调拒绝发布。需要根据 TP/PP/EP 布局明确各域负责的文件以及合并规则。当前实现要求 `EP=1`，且同一文件的所有域使用相同的有序名单。 |
| 临终文件必须是 `torch_dist`，或要求恢复每个故障 rank 的独立 RNG | TFT 26.2.0 的 `save_info` 只提供当前 rank 的域和选中成员。前者需要额外的全局保存规划，后者需要原 rank 的状态来源。当前临终文件使用 legacy 格式，只保存原生 reader 所需的单份 RNG 记录。 |
| ARF 缺少原始参数组键或完成步进度 | 参数组键用于创建替换进程的原生优化器，缺失时替换进程初始化失败。优化器已建立但缺少原 rank 进度时，按 TFT 下发的磁盘动作恢复，磁盘加载需要全 world 参与。 |
| ARF 使用自定义 launcher 或模型初始化流程 | 设备重新接入、相同逻辑 rank 和 rendezvous 由部署层负责。客户需提供 `optimizer_group_keys` 所需的启动交换结果，或依据固定拓扑证明可重建相同结果。其他依赖集合通信返回值的启动操作也要核对。 |
| 使用外部数据流或持久 worker | 当前回退路径使用 Megatron 原生普通 DataLoader。扩展时需要客户提供可按原消费游标定位的数据源，并恢复对应的 worker 状态。 |
| 客户 fork 的多节点训练与恢复 | 现场需确认 TorchNPU/HCCL/CANN、TFT wheel、启动器、拓扑和 checkpoint 格式的组合。临终读回、UCE、ARF 及性能目前均待客户环境验证。 |
