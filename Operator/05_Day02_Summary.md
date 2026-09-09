# Day 2 总结：CUDA Execution Model 与第一个 Kernel

## Day 2 学习目标

Day 2 的目标不是写复杂 CUDA，而是建立一条完整主线：

> 一个算子如何被拆成 Threads，这些 Threads 如何组成 Blocks 和 Warps、被调度到 SM，并最终完成数据读取、计算和写回。

## 1. 软件执行层级

```text
Grid
└── Blocks
    └── Warps
        └── Threads
```

- Grid：一次 Kernel Launch 产生的全部 Blocks；
- Block：Threads 的协作单位，也是调度到 SM 的单位；
- Warp：NVIDIA GPU 的核心执行与调度粒度，1 Warp = 32 Threads；
- Thread：逻辑执行单位，通常负责一小份数据。

## 2. 软件结构如何落到硬件

```text
Kernel Launch
     ↓
Grid 中产生许多 Blocks
     ↓
Blocks 被分配到多个 SM
     ↓
每个 Block 被拆成多个 Warps
     ↓
Warp Scheduler 选择 Ready Warp
     ↓
Warp 中的 Threads 执行同类指令、处理不同数据
```

需要避免两个误解：

- Thread 不固定绑定某个 CUDA Core；
- Block 不与 SM 一一对应。一个 Block 只会在一个 SM 上执行，但一个 SM 可以驻留多个 Blocks。

## 3. Warp Scheduling 与 Latency Hiding

当某个 Warp 因等待内存而暂时不能继续时，SM 可以调度其他 Ready Warp。

```text
Warp A 等待内存
        ↓
调度 Warp B / C / D
        ↓
尽量让执行单元保持忙碌
```

这不会缩短内存访问本身的延迟，而是用其他工作覆盖等待时间。

资源使用会影响这一能力。例如每 Thread 使用过多 Register，会减少可驻留 Warps，进而降低 Latency Hiding 能力。

## 4. Memory Hierarchy 的初步直觉

| 存储 | 范围 | 核心直觉 |
|---|---|---|
| HBM | GPU 大数据 | 容量大、距离远、搬运昂贵 |
| Shared Memory | 同一 Block | 高速共享工作台，适合数据复用 |
| Register | 单个 Thread | 极快、私有、容量很小 |

Shared Memory 的价值来自 Data Reuse，而不是“它比 HBM 快”这一点本身。数据只用一次时，额外搬运通常不划算。

## 5. Vector Add 串起全部概念

```cpp
__global__ void add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}
```

逐层解释：

- `__global__`：这是可由 Host 发起的 GPU Kernel；
- `a`、`b` 是只读输入，`c` 是输出，`n` 是有效元素数；
- `i`：当前 Thread 对应的全局数据索引；
- `i < n`：防止向上取整后多出的 Threads 越界访问；
- `c[i] = a[i] + b[i]`：当前 Thread 真正执行的 Operator Computation。

## 6. 完整数据与执行流程

```text
CPU 决定 Blocks 和 Threads / Block
                  ↓
Launch Kernel
                  ↓
Grid → Blocks → 分配到 SM
                  ↓
Block → Warps，1 Warp = 32 Threads
                  ↓
Warp Scheduler 调度 Ready Warps
                  ↓
Thread 计算全局索引 i
                  ↓
从 HBM Load A[i]、B[i]
                  ↓
执行 Add
                  ↓
Store C[i] 到 HBM
```

## 7. 本日关键纠正

### “Shared Memory 是 SM 内所有 Threads 共享”

不准确。Shared Memory 的共享与协作范围是同一个 Block。

### “Register 太多会增加 Memory Latency”

不准确。Memory Latency 本身没有因此变长；可驻留 Warps 变少，使隐藏 Latency 的能力下降。

### “每个 Block 对应一个 SM”

不准确。Block 被调度到某个 SM；一个 SM 可以同时驻留多个 Blocks。

### “`int i = ...` 只是算 Thread ID”

不够准确。它的实际目的，是计算当前 Thread 负责的全局数据索引。

### “为什么 `i` 会越界”

因为 Blocks 必须按完整大小启动。若 `N` 不是 Block Size 的整数倍，最后一个 Block 会包含多余 Threads，因此需要 `if (i < N)`。

## 8. 学习者已经掌握的内容

- 能计算 `256 Threads = 8 Warps`；
- 能解释其他 Ready Warps 如何覆盖某个 Warp 的等待；
- 能区分 HBM、Shared Memory 与 Register 的基本角色；
- 能从 Data Reuse 判断 Vector Add 通常不需要 Shared Memory；
- 能逐行解释基础 CUDA Vector Add Kernel；
- 能理解逻辑 Threads 数量远多于物理计算单元，并由 SM 分批调度执行。

## 9. Day 2 最小记忆集

> 32 Threads = 1 Warp。Block 是协作与调度边界，会被放到某个 SM；一个 SM 可以驻留多个 Blocks。SM 调度 Ready Warps 来隐藏等待。HBM 大但远，Shared Memory 供 Block 内复用，Register 由 Thread 私有。Kernel 中先看 Mapping，再看 Computation、Memory Access、Data Reuse 和资源约束。

## 下一步

Day 3 可继续学习更具体的内存访问问题，例如 Memory Coalescing，并逐步补充 L1/L2、Shared Memory Bank Conflict 等概念。
