# MindSpeed-LLM 26.2.0 HA 逐函数迁移

客户在现有训练分支接入时，可从[适配指南](tft-26.2-adaptation-guide-zh.md)查看补丁用法、配置和各项修改的原因。

实现来源是 Ascend/MindSpeed-LLM `26.2.0@effa8e78` 默认树，目标是 Megatron
`core_v0.18.0@ba7b5ebc`。TFT 协议依据是 MindIO `v26.2.0@6cb6683c`。适配文件置于
目标 Megatron 的 `megatron/training/high_availability`。原有独立 HA 工作树用于对照。

以下路径中的 `core/HA` 和 `tasks/HA` 分别指官方默认树的
`mindspeed_llm/core/high_availability` 和 `mindspeed_llm/tasks/high_availability`。

| 官方文件及函数 | 保留的行为与实现 | 旧优化器调用及替换必要性 | 目标位置 |
| --- | --- | --- | --- |
| `tasks/HA/high_availability_helper.py::get_env_args` | 逗号分隔且大小写不敏感，`dump`、`retry`、`recover` 保留各自开关语义 | 去掉范围以外的 ZIT 分支，原生训练无 LLM feature 注册器 | 同名模块，初始化入口 |
| `tasks/HA/initialize_patch.py::initialize_distributed_wrapper` | 先绑设备，再启 TFT，随后初始化 world 和模型并行组 | 取消 `ttp_initialize_replica_dp_group` 的自建副本组，读取目标 native groups | `initialize.py::_initialize_distributed`，`initialize_patch.py` |
| `core/HA/tft_train_initialize.py::tft_init_controller_processor` | `TTP_ADDR` 回退，`CONTROLLER_ADDR`、`PROCESSOR_ADDR`，8000，MindX 外置 controller，无 MindX 时 rank 0 本地 controller，随后启动 processor 并读取 reboot | 去掉 ZIT，目标 SDK 不支持将框架名设为 Megatron，使用通用 PyTorch 路径和原来的 stop/clean 开关 | `tft_train_initialize.py` |
| 同文件 `tft_register_processor` | 先上报 optimizer 副本和 plain DP，再注册 save → rename → stop → clean → repair → rollback → rebuild → stream sync | `ttp_get_replica_dp_num` 的固定副本数换成 native 实例数；保留 Dense、Expert 域编号 | `tft_train_initialize.py`、`tft_replica_group.py` |
| 同文件 `tft_train/tft_train_wrapper` | `tft_exception_handler` 包装原生训练，替换 rank 在重建后报告 `ARF FINISH` | 去掉写死的 LLM trainer、框架名和位置下标，传入显式训练上下文 | `training.py::pretrain`、`tft_train_initialize.py` |
| `core/HA/tft_replica_group.py::build_repair_group/get_repair_group/destroy_repair_group` | 根据 TFT `rank_list` 创建临时 HCCL、Gloo 组及释放顺序 | 常驻的 `build_dp_cp_replica_group/build_dp_ep_replica_group` 改用 native intra/inter；controller 会排序，上报前检查实际 shard 同余关系 | `tft_replica_group.py` |
| `core/HA/tft_replica_optimizer*.py` 与 `tft_replica_chainedoptimizer.py` 的 step；官方 `training.py::train` | 保留 TFT 更新起止和 pause 的顺序 | 旧 `current_step`、独立 leaf step 和 buffer cache 换成顶层 optimizer 事务。设备完成、scheduler 和实际样本计数完成后才提交 | `training.py::train_step/train`、`tft_train_initialize.py` |
| `core/HA/tft_dump.py::tft_save_callback/get_checkpoint_name/save_checkpoint/tft_rename_callback` | 验证 TFT 选人，列表首项 writer，临时目录及 TFT 通知发布 | `set_dump_args/need_write_file/sync_gather_all_model_params` 和旧 `save_parameter_state` 绑定旧副本组。按 native bucket 与实际顺序收集，写目标 legacy 文件再更新 tracker | `tft_dump.py`、`native_distopt.py` |
| `core/HA/tft_stop_clean.py::stop_callback/clean_callback/model_update_npu_tensor_to_safe/unset_gather_handle/torch_sync` | stop，分级 clean，按 retry 恢复 HCCL，重启设备；清理 DDP 句柄 | `optim.update_npu_tensor_to_safe()` 改为遍历 native leaf，访问原生 DDP buffer 与 optimizer 键 | `tft_stop_clean.py`、`tft_tensor_recovery.py` |
| `core/HA/tft_optimizer_data_repair.py::repair_callback/send_rank_repair/recv_rank_repair/save_and_send_ckpt/recv_ckpt_from_peer` | 由 TFT repair_type 分派，建立临时组，先传内存元数据，再传 optimizer shard。沿用大小、序列化张量和 point-to-point 次序 | `state_dict_memory/state_dict_by_idx/load_state_dict_memory` 及 `send_optim_param_state/recv_and_load_optim_param_state` 改为实际 leaf、native range 和状态访问；验收目标 rank 的本地进度 | `tft_optimizer_data_repair.py`、`native_distopt.py` |
| 同文件 `load_ckpt_repair` | 只响应两个 TFT 磁盘动作，走原生 checkpoint loader 并汇报实际加载步 | 移除旧精度工具的 tracker 改写；固定版原生 loader 集合通信需要整 world 一致 | `tft_optimizer_data_repair.py` |
| `core/HA/tft_rollback.py::rollback_callback/feature_rollback/build_dataset/rebuild_global_vars/gather_model_params_from_optimizer`；`tasks/HA/dataloader_patch.py` | 保留清理、copy → gather、迭代器重建、同步、释放修复组及恢复训练位置的顺序 | `_copy_main_params_to_model_params` 按 native leaf，`sync_gather_all_model_params` 改为原生 DDP `start_param_sync(force_sync=True)`。恢复已有 scheduler 前复位步数；0.18 用原生 builder 重建 iterator | `tft_rollback.py`、`dataloader_patch.py` |
| `tasks/HA/communication_patch.py::*wrapper`、`core/HA/tft_arf_group_repair.py::update_arf_reboot_flag/arf_rebuild_process_group_callback/init_* /update_model_and_optim_related_group` | 保留本地先启动、保存和还原 load 路径、TFT 重建通知、world → 并行组 → optimizer 组 → 句柄重绑 | 0.18 的组成员来自 native group 创建，Gloo 是否重建与旧句柄非空无关。使用 TorchNPU 原句柄重连 HCCL。原生 optimizer 启动的返回值不能用旧 wrapper 跳过 | `initialize_patch.py`、`communication_patch.py`、`tft_arf_group_repair.py` |

## 正式接口与部署边界

- 旧 SDK 的 `save_info` 给出当前 rank 的域和选中成员。它不提供全局保存计划或失败
  rank 的 RNG。native legacy reader 的默认 RNG 格式、同一文件多个域的所有权以及
  `torch_dist` 故障保存需分别核对。只在能证明完整覆盖时发布。
- 该 TFT 接口不传目标 rank 的专属 RNG 和数据源状态。UCE 可用本进程已完成步的小状态。
  ARF 替换进程需要客户已有的原 rank 状态，或由 TFT 下发磁盘恢复动作。现有 donor
  的 rank 状态不能证明是目标状态。
- ARF 启动的 native optimizer 参数组键须来自客户已有的启动元数据，或从客户固定拓扑
  证明可唯一推导。设备重新接入和原 logical rank rendezvous 仍由部署层负责。
- 官方版本是源码来源。客户 fork、HCCL/TorchNPU/CANN、wheel、原 rank 启动语义、
  训练拓扑及 checkpoint 格式的实际运行效果均待客户环境确认。

## 接入与恢复条件

- 适配入口保留 `HIGH_AVAILABILITY` 的独立开关。临终保存使用 `dump`，UCE 使用
  `retry`，同时启用 UCE 和 ARF 使用 `retry,recover`。按原版使用 `TTP_ADDR`、
  `CONTROLLER_ADDR`、`PROCESSOR_ADDR` 和端口 8000。设置 `MINDX_TASK_ID` 时
  使用外置 controller。需替换 rank 0 的 ARF 部署应让 controller 独立于训练进程。
- 当前代码面向原生 BF16 Adam、DDP、固定 world size、Gloo 可用、至少两个原生 optimizer
  实例的训练。启动参数包含 `--use-distributed-optimizer`、
  `--num-distributed-optimizer-instances 2`、`--rerun-mode disabled`、
  `--num-workers 0` 和 `--save`。周期 checkpoint 可保持客户原生格式。
  现有 Ascend kernel、设备适配和 TorchNPU 原位重连能力由客户部署提供。
- 官方 wheel 的 VERSION 构建脚本记录组件路径最后一次变更，`6cb6683c` tag 对应
  `9a28823c002d113d109c369e4faf8790ddcd8d0a` 的组件树。
- 临终目录继续使用临时目录与 rename。列表首项是 TFT writer，固定 SDK 的逻辑
  shard 选择使其同时覆盖 shard 0。汇聚中按进程组实际顺序重新定位各 shard。
  使用 `torch_dist` 周期 checkpoint 时，legacy 临终文件落到 `<save>.tft`，恢复
  时用 `--ckpt-format torch --load <save>.tft`。reader 的 DP-zero scatter
  只传 master weights 和 moments，因此目标 `distrib_optimizer.py` 保留元数据已
  恢复的 Adam 标量 step。采用 native legacy singleton RNG 语义的场景可以使用此盘格式。
- 目标 rank 的已完成步进度保存在本进程内存。需要跨进程恢复时，`pretrain` 的可选
  `tft_recovery` 参数可接客户已有的数据来源。其 `rank_progress(rank, step)`
  返回原 rank 的 `rank`、`step`、`counters`、`flops`、`scheduler`、
  `rng` 和 `model_buffers`。键与 `TrainingState._progress` 一致。
  `optimizer_group_keys(rank, exchange)` 返回原生 optimizer 工厂该次 world
  `all_gather_object` 的逐 rank 结果。替换进程核对原 rank 的本地键后使用该结果。
  适配器没有跨进程进度存储或新的 TFT 指令。
- UCE 磁盘回退接受全 world 的 TFT 磁盘动作，调用原生 loader 并汇报实际步。网内
  修复保留官方的 metadata、optimizer tensors、日志三阶段顺序。native logger
  在完成步已经执行，回退记录已收到的日志值，并重置 timer 和 rerun 状态。

## 当前未闭合的组合

| 条件 | 当前处理 |
| --- | --- |
| 同一 legacy 文件的 Dense、Expert 域选出不同 writer，或 EP 大于 1 | 回调拒绝发布。需结合客户 TP/PP/EP 布局和保存格式定义各域文件的合并规则。 |
| 要求故障保存仍写 `torch_dist`，或需要精确恢复失败 rank 的独立 RNG | 固定版 `save_info` 仅提供各参与者的域及名单。需要可核验的保存规划或客户已有的 rank 状态来源。 |
| ARF 缺少 optimizer 键交换结果或原 rank 专属进度 | 键交换结果是原生建 optimizer 的启动依赖。没有它会拒绝替换进程启动；只有专属进度缺失时，可由 TFT 正式选择磁盘动作。 |
| 外部或带持久 worker 的数据流 | 需要客户提供与原消费游标一致的数据定位与 worker 状态恢复。当前端到端回退走原生普通 DataLoader。 |
| 多节点、特定 Ascend wheel、HCCL 重连和性能 | 代码依据固定官方源码迁移。临终读回、UCE、ARF 的现场运行均待验证。 |
