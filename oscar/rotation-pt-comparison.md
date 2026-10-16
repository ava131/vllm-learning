# OSCAR：本地 Ascend 版本与上游 SGLang 的启动和旋转矩阵对照

> 源码快照：2026-10-16。本地 `gpt_new_oscar_kimi` 分支 `codex/aisbench-stable`，提交 `f96820f`；上游 [FutureMLS-Lab/OSCAR `sglang-rebase`](https://github.com/FutureMLS-Lab/OSCAR/tree/339368e8fb440c53d5e157a5974036f25f3841a0)，提交 `339368e`。本地目标是 **Qwen3.5-27B-w8a8-mtp / Ascend / TP4**，上游所比较的运行脚本针对 **Qwen3.8-27B / SGLang / TP2**。这是源码与论文对照，不是两套模型的质量或性能实测对比。

## 先看结论

1. **启动方式不同。**本地一键入口先安装、编译、生成或校验旋转 `.pt`，然后启动服务；不要求事先准备 `.pt`。上游 `rotation/run/qwen3.8-27b.sh` 默认先检查已有的 K/V 两个 `.pt`，缺失便报错；只有显式 `STARTUP_CALIBRATE=1`，才允许缺文件进入服务内的启动校准流程。
2. **默认旋转不同。**本地目标 FULL attention 层默认使用固定、无需样本的 Hadamard 矩阵 `Rk = Rv = H`。上游若启动校准，则用 Q/K/V 样本统计构造 `Rk = U_Q · H · Pbr`、`Rv = U_S · H · Pbr`。Hadamard 是论文完整旋转的一部分；纯 `H` 不是论文的完整校准方案。
3. **产物不兼容。**本地将 K/V 放进一个带模型指纹的 `hadamard.pt`；上游用按全局层号索引的 K/V 两个文件。即便选相同的纯 Hadamard 矩阵，也不能直接把一个项目的 `.pt` 给另一个项目加载。
4. **裁剪设置也不同。**本地目标配置未指定 `k_clip_ratio` / `v_clip_ratio`，生产调用默认两者为 `0.0`（不裁剪）；上游 Qwen3.8 评测启动器传入非零的 K/V 比例 `0.96` / `0.92`。因此即使旋转矩阵相同，整体 INT2 路径仍不是同一数值配置。

## 1. `.pt` 是什么

这里的 `.pt` 保存各注意力层使用的正交旋转矩阵，**不是模型权重，也不是运行中的 KV cache**。K 在量化写入前旋转；Q 使用配套的 K 旋转参与历史注意力；V 使用 V 旋转，输出再按逆旋转回到原坐标。正交旋转在不量化的理想数学情形下保留注意力语义，其实际用途是改变 INT2 量化面对的坐标分布、减少有影响的误差。参见[论文第 3 节公式 (3)](https://arxiv.org/html/2605.17757v1)。

## 2. 启动时如何取得 `.pt`

| 步骤 | 本地 Ascend 版本 | 上游 `qwen3.8-27b.sh` |
| --- | --- | --- |
| 入口 | [`scripts/install_serve.sh`](https://gitcode.com/jpl123/gpt_new_oscar_kimi/blob/f96820f/scripts/install_serve.sh) 调用 `tools.install_serve`；阶段顺序包含安装、编译、`prepare-rotations`、正式服务。该入口不运行 probe。 | [模型脚本](https://github.com/FutureMLS-Lab/OSCAR/blob/339368e8fb440c53d5e157a5974036f25f3841a0/rotation/run/qwen3.8-27b.sh#L8) 调用共用 `need_rot` 后再 `launch`。脚本自身不拟合矩阵。 |
| 默认缺文件 | [`tools/prepare_rotations.py`](https://gitcode.com/jpl123/gpt_new_oscar_kimi/blob/f96820f/tools/prepare_rotations.py#L43) 读取目标模型 `config.json`，确定 FULL 层及维度，在 NPU 上生成固定 Hadamard 并保存；已有文件则严格校验和复用。 | [`need_rot`](https://github.com/FutureMLS-Lab/OSCAR/blob/339368e8fb440c53d5e157a5974036f25f3841a0/rotation/run/_common.sh#L17) 从预置镜像或 `ROT_DIR` 指定的 RotationZoo 目录找唯一的 K/V 文件；缺失就退出。 |
| 显式启动校准 | 当前自动入口不支持；`rotation_method` 不是 `hadamard` 会报错。本地虽有 `build_sample_artifact` 接口，但启动流程没有采样 worker 或经确认的 NPU 特征分解器接入。 | 设 `STARTUP_CALIBRATE=1` 时，`need_rot` 允许缺文件并指定输出名。服务在一对文件缺失或发布未完成时运行校准：先用 identity，执行校准 prompts，收集 Q/K/V，拟合并发布两个 `.pt`。若一对文件已完整存在，单设此开关不会无条件重算。见[路径决策](https://github.com/FutureMLS-Lab/OSCAR/blob/339368e8fb440c53d5e157a5974036f25f3841a0/python/sglang/srt/mem_cache/oscar_rotation_paths.py#L117)及[校准入口](https://github.com/FutureMLS-Lab/OSCAR/blob/339368e8fb440c53d5e157a5974036f25f3841a0/python/sglang/srt/entrypoints/oscar_startup_calibration.py#L235)。 |
| 另一条生成途径 | 无自动离线 Q/K/V dump 生成流程。 | 可另行用 [`rotation/compute_kv_rotation.py`](https://github.com/FutureMLS-Lab/OSCAR/blob/339368e8fb440c53d5e157a5974036f25f3841a0/rotation/compute_kv_rotation.py#L350) 从已采集的 Q/K/V dump 离线生成，也可显式选择纯 `hadamard`；这不是 `qwen3.8-27b.sh` 默认执行的步骤。 |

本地生成路径是 `artifacts/rotations/<模型指纹>/hadamard.pt`。这里的指纹由模型配置文件和存在时的 `model.safetensors.index.json` 内容计算，**没有逐字节哈希所有权重张量**。详见[`model_geometry`](https://gitcode.com/jpl123/gpt_new_oscar_kimi/blob/f96820f/tools/prepare_rotations.py#L14)。

## 3. 旋转矩阵怎样计算

| 项目 | 本地默认 | 上游启动校准 |
| --- | --- | --- |
| 数据来源 | 不读取校准数据或模型 Q/K/V；只需 head dimension 与 FULL 层集合。 | 默认可使用 GPQA-Diamond prompts，也可指定其他校准输入；服务收集模型实际 Q/K/V。详见[输入准备](https://github.com/FutureMLS-Lab/OSCAR/blob/339368e8fb440c53d5e157a5974036f25f3841a0/python/sglang/srt/entrypoints/oscar_startup_calibration.py#L74)。 |
| 数学形式 | 每个目标 FULL 层 `Rk = Rv = H`；同维度的固定 Sylvester-Hadamard 数值相同。MTP draft 层另用显式 identity，不在目标 FULL 层 `.pt` 内。详见[`build_hadamard_artifact`](https://gitcode.com/jpl123/gpt_new_oscar_kimi/blob/f96820f/oscar_ascend/rotations.py#L183)与[运行时](https://gitcode.com/jpl123/gpt_new_oscar_kimi/blob/f96820f/oscar_ascend/runtime.py#L668)。 | Q 相关协方差决定 K 的特征基 `U_Q`，K/Q 加权的 V 统计决定 `U_S`；特征分解后分别组成 `U_Q·H·Pbr`、`U_S·H·Pbr`。K/V 通常不同，也随权重、数据和校准设置变化。详见[统计收集](https://github.com/FutureMLS-Lab/OSCAR/blob/339368e8fb440c53d5e157a5974036f25f3841a0/python/sglang/srt/mem_cache/oscar_calibration.py#L198)与[拟合](https://github.com/FutureMLS-Lab/OSCAR/blob/339368e8fb440c53d5e157a5974036f25f3841a0/python/sglang/srt/mem_cache/oscar_calibration.py#L650)。 |
| 数值生成 | 固定 H 在 NPU 上以 FP32 构造、验证并保存。 | 在线校准的协方差和特征分解使用 FP64，最终旋转以 FP32 存储；离线脚本也支持不同旋转目标和组合。 |
| 旋转后的裁剪 | [`configs/target.json`](https://gitcode.com/jpl123/gpt_new_oscar_kimi/blob/f96820f/configs/target.json) 未指定 K/V clip；[生产 store 调用](https://gitcode.com/jpl123/gpt_new_oscar_kimi/blob/f96820f/oscar_ascend/integration/cache_storage_ops.py#L58) 默认 `0.0`，即不裁剪。 | [Qwen3.8 共用评测启动器](https://github.com/FutureMLS-Lab/OSCAR/blob/339368e8fb440c53d5e157a5974036f25f3841a0/rotation/eval_oscar_gpqa.sh#L345) 默认传入 K `0.96`、V `0.92`；这与矩阵拟合是两个独立设置。 |

**如果上游明确选择纯 `hadamard`：**相同维度、相同 Sylvester 排列时，两边矩阵在数学上相同；例如 `D=256` 的元素均为 `±1/16`。这不意味着序列化后的 `.pt` 逐字节相同，也不意味着上游默认选用纯 H。特别是同结构但不同权重的模型，校准得到的 `U_Q`、`U_S` 不应预期相同。

## 4. 文件结构和复用边界

| | 本地 | 上游 |
| --- | --- | --- |
| 文件 | 一个 `hadamard.pt` | 一对 `k_rotation_*.pt`、`v_rotation_*.pt` |
| 层键及矩阵字段 | 字符串层名，如 `model.layers.3.self_attn.attn`；同一层内有 `Rk`、`Rv` | 整数全局层号；K/V 各自文件的层条目含 `rotation`，校准文件还含特征值及校准元信息 |
| 校验/发布 | 模型与 PR 指纹、层集合、维度、FP32、有限性、正交性校验；单文件临时保存后替换 | 在线路径对 K/V 一对文件加锁、临时写入、校验并用 pending/complete 标记发布；加载器读取另一种 schema |

参见本地[`save_artifact` / `load_artifact`](https://gitcode.com/jpl123/gpt_new_oscar_kimi/blob/f96820f/oscar_ascend/rotations.py#L324)，以及上游[检查点结构与发布](https://github.com/FutureMLS-Lab/OSCAR/blob/339368e8fb440c53d5e157a5974036f25f3841a0/python/sglang/srt/mem_cache/oscar_calibration.py#L710)和[加载器](https://github.com/FutureMLS-Lab/OSCAR/blob/339368e8fb440c53d5e157a5974036f25f3841a0/python/sglang/srt/mem_cache/memory_pool.py#L282)。

## 5. 论文证据与当前验收边界

[论文第 3 节公式 (3)](https://arxiv.org/html/2605.17757v1)给出完整的 `U·H·Pbr` 旋转；[图 2](https://arxiv.org/html/2605.17757v1)把纯 Hadamard 作为对照，显示其在论文测试的 INT2 场景中仍有明显注意力及输出误差。[第 5.3 节表 4](https://arxiv.org/html/2605.17757v1)在 Qwen3-8B、相同 INT2 位数与 sink/recent 保护下，报告完整 `U·H·Pbr` 五项任务平均分 **70.01**，去掉 `U`、保留 `H·Pbr` 为 **32.82**。后一行是 `H·Pbr`，不是严格的纯 H。

因此本地默认旋转只是论文中的 **data-free Hadamard 对照形式**，不能将论文完整 OSCAR 的质量结果归到这个 `.pt`。论文结果也不是本地 Qwen3.5-27B-w8a8-mtp / Ascend 的实测分数；本文件没有生成矩阵数值 diff、模型质量验收或性能验收证据。
