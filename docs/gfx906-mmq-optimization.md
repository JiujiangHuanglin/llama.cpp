# gfx906 MMQ 优化实验记录

## 背景

- 硬件：2× AMD Instinct MI60/MI50 (gfx906), ROCm 10.2
- 模型：Qwen3.8-27B Q8_0 (27.41 GiB)
- 配置：`-ngl 999 --split-mode layer -b 4096 -ub 4096 -fa on`
- 基线版本：llama.cpp b11290
- 分支：`fattn-tile-skip-continue`

## Profiler 数据（32K Prefill）

| Kernel | 占比 | Avg | 说明 |
|---|---|---|---|
| `mul_mat_q` | 69.7% | 27.1 ms | Q8_0 权重矩阵乘，Prefill 主导 |
| `flash_attn_tile` | 16.7% | 201 ms | 长上下文 attention |
| `gated_delta_net_cuda` | 7.8% | 31.4 ms | SSM 层 |
| 其余 | <3% | — | — |

## MMQ_DEBUG 诊断

在 `mmq.cuh` 的 `mul_mat_q_switch_J` 中加打印，实际运行输出：

[MMQ] type=8 J_best=128 ntiles_J_best=32 ncols_opt=4096 smpbo=65536

- `type=8` = Q8_0
- `J_best=128` = 4096 batch 下走 J=128 配置
- `smpbo=65536` = 64 KB LDS 上限

## 已做实验

### 实验 1：nthreads 512 → 256（失败，已回滚）

- 改动：`mmq-config-gcn.cuh` 中 Q8_0 J=128 行的 nthreads 从 512 改 256
- 结果：**366.5 → 275.4 t/s (-25%)**
- 原因：nwarps 从 8 降到 4 后，每 warp 处理的 j 值翻倍（16→32），VGPR 压力翻倍导致 spill
- **结论：nwarps=8 是 gfx906 的寄存器平衡点，不可降**

### 实验 2：stream_k = true（保留）

- 改动：Q8_0 J=128 行的 `stream_k_` 从 false 改 true
- 结果：**360.82 → 365.00 t/s (+1.16%)**
- 16K Prefill，`-r 3` 单次测量
- **结论：边际改善，保留**

### 实验 3：I 调优（未做）

- 理论：LDS 从 33 KB 降到 16.6 KB，occupancy 提升
- 代价：dispatch 次数翻倍
- 预计：大概率变慢，暂不实施

## 参考 fork

[iacopPBK/llama.cpp-gfx906](https://github.com/iacopPBK/llama.cpp-gfx906)

- 基线：b58fd8306（比本 fork 老约 5000 commit）
- 核心：`gfx906/` 专用 kernel 目录，包含：
  - `fused/gather-q8.cu`、`fused/norm-fused-q8.cu`
  - `attention/fattn-q8.cu` + 多个 dkq/dv 实例
  - `quantize/q8-cache.cuh`（703 行，Q8_1 激活缓存）
- **结论：不能直接移植**，接口和目录结构完全不同
- `q8-cache.cuh` 优化的是 `quantize_mmq_q8_1`（占 1%），不是 `mul_mat_q`（占 69.7%）

## 最终状态

MMQ 侧能做的优化已到极限：
- 配置表参数（nthreads / I / J）已调过
- stream_k 已启用
- gfx906 无 MFMA，dp4a 路径的算力上限是硬约束

**下一步优化方向：`flash_attn_tile`**（占 16.7% @32K，随序列线性增长）

## 调试方法

启用 MMQ 调度打印：

```bash
MMQ_DEBUG=1 ./bin/llama-bench -m model.gguf ...
运行时输出：
[MMQ] type=8 J_best=128 ntiles_J_best=32 ncols_opt=4096 smpbo=65536
关键 commit
commit	内容
fb6918c31	baseline（含 FA skip-continue 优化）
bfdbb0a69	MMQ Q8_0 J=128 stream_k (+1.2%)
(待提交)	MMQ_DEBUG 诊断
