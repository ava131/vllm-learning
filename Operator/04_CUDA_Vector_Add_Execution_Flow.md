# Day 2 · Part 4：CUDA Vector Add 完整执行流程

## 1. Kernel 代码

```cpp
__global__ void vector_add(
    const float* A,
    const float* B,
    float* C,
    int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        C[i] = A[i] + B[i];
    }
}
```

## 2. 逐行理解

### `__global__`

表示这是一个 GPU Kernel，可以从 CPU（Host）侧发起 Launch。

### 参数

```cpp
const float* A,
const float* B,
float* C,
int N
```

- `A`、`B` 指向输入数组；`const` 表示 Kernel 只读它们；
- `C` 指向输出数组，因此没有 `const`；
- `N` 是数组实际包含的元素数量。

### 全局数据索引

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

含义：

```text
前面完整 Blocks 包含的 Threads 数
+ 当前 Thread 在 Block 内的位置
= 当前 Thread 负责的全局数据索引 i
```

例如 Block 3、每 Block 256 Threads、Block 内 Thread 10：

```text
i = 3 × 256 + 10 = 778
```

这个 Thread 负责 `A[778]`、`B[778]` 和 `C[778]`。

### 边界检查

```cpp
if (i < N)
```

Kernel 启动的线程数通常会向上取整到完整 Block，因此 `i` 可能大于或等于 `N`。

例如：

```text
N = 1000
Threads / Block = 256
Blocks = ceil(1000 / 256) = 4
总 Threads = 4 × 256 = 1024
```

最后 24 个 Threads 的 `i` 为 `1000~1023`，已经超出数组有效范围。`if (i < N)` 会让它们不执行数组访问，避免越界。

### 真正的 Operator 计算

```cpp
C[i] = A[i] + B[i];
```

它回答“每个 Thread 真正计算什么”。而 `int i = ...` 回答“这个 Thread 负责哪一份数据”。

```text
Parallel Mapping：谁算哪份数据？
Operator Computation：每份数据怎么算？
```

## 3. CPU 侧如何 Launch

```cpp
int threads = 256;
int blocks = (N + threads - 1) / threads;

vector_add<<<blocks, threads>>>(A, B, C, N);
```

- `threads = 256`：每个 Block 有 256 Threads，即 8 Warps；
- `(N + threads - 1) / threads`：整数方式向上取整；
- `<<<blocks, threads>>>`：指定 Grid 中的 Block 数量和每 Block 的 Thread 数量。

## 4. 从 Launch 到执行的完整流程

```text
CPU / Host 发起 Kernel Launch
              ↓
建立一个包含许多 Blocks 的 Grid
              ↓
Blocks 被动态调度到多个 SM
              ↓
每个 Block 拆成 Warps
              ↓
256 Threads / Block = 8 Warps / Block
              ↓
SM 的 Warp Scheduler 选择 Ready Warp
              ↓
每个 Thread 计算自己的全局数据索引 i
              ↓
从 HBM 读取 A[i]、B[i]
              ↓
执行加法
              ↓
将 C[i] 写回 HBM
```

### 关键纠正：Block 与 SM 不是一一对应

学习者原理解：

> 每个 Block 对应一个 SM。

更准确的说法：

- 一个 Block 会被调度到某一个 SM，执行期间不会被拆到多个 SM；
- 一个 SM 可以同时驻留多个 Block，具体数量受 Register、Shared Memory、Warp/Thread 上限等资源约束；
- Block 内的 Threads 被组成 Warps，由 SM 调度 Ready Warps 执行。

## 5. Memory 路径

对某一个 Thread，可以先粗略理解为：

```text
HBM: A[i], B[i]
        ↓ Load
Register / Execution Path
        ↓ Add
Register 中的结果
        ↓ Store
HBM: C[i]
```

这里通常不使用 Shared Memory，因为 `A[i]` 和 `B[i]` 基本只用一次，没有明显的数据复用。

## 6. 学习者问答与关键纠正

### Q1：`int i = ...` 与 `C[i] = A[i] + B[i]` 分别解决什么？

学习者回答：前者解决“哪个线程”，后者解决“每个线程怎么算”。

更精确地说：

- `int i = ...` 计算当前 Thread 对应的全局数据索引，解决谁处理哪份数据；
- `C[i] = A[i] + B[i]` 定义该 Thread 对这份数据执行的计算。

### Q2：为什么 Vector Add 通常不需要 Shared Memory？

因为每个 `A[i]`、`B[i]` 通常只被一个 Thread 使用一次。先搬入 Shared Memory 不会产生复用收益，反而增加搬运和同步成本。

### Q3：一百万个 Threads 如何在数量有限的计算单元上执行？

Grid 被分成许多 Blocks；Blocks 被逐步调度到各个 SM；每个 Block 又被分成 Warps；Warp Scheduler 从 Ready Warps 中选择可执行的 Warp。Threads 是逻辑并行单位，并不意味着 GPU 同时拥有一百万个独立物理计算单元。

### Q4：为什么必须有 `i < N`？`i` 真的可能大于 `N` 吗？

会。更准确地说，`i` 可能大于或等于 `N`。因为 Block 数需要向上取整，最后一个 Block 中可能包含多余 Threads。边界检查负责屏蔽它们，防止访问 `A[N]`、`B[N]`、`C[N]` 及更后面的非法位置。

## 7. 阅读基础 Kernel 的五步方法

1. Thread / Block 如何 Mapping？
2. 每个 Thread 具体计算什么？
3. 数据从哪里 Load、向哪里 Store？
4. 数据是否会被重复使用？
5. 是否存在资源、调度或分支问题？

## 本节最小记忆集

> `i` 决定当前 Thread 处理哪一个元素；核心表达式决定它怎么算；`i < N` 屏蔽向上取整后多出的 Threads。Blocks 被调度到 SM，Blocks 再拆成 Warps，SM 调度 Ready Warps 完成 Load、Compute、Store。
