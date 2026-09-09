# Day 2 · Part 3：HBM、Shared Memory 与 Register

## 本节目标

先建立一个简化但实用的 GPU 存储直觉：

> 大 Tensor 通常放在 HBM；需要协作复用的数据可以放进 Shared Memory；单个 Thread 正在使用的小数据通常放在 Register。

本节暂不展开 L1/L2 Cache、Memory Coalescing、Bank Conflict 等细节。

## 1. HBM：大、远、数据搬运昂贵

HBM（High Bandwidth Memory）是 GPU 的高带宽显存。输入、输出等大 Tensor 通常存放在这里。

```text
            GPU
┌────────────────────────────┐
│     SM      SM      SM     │
│       Compute Units        │
└────────────────────────────┘
             ↕
            HBM
       A / B / C Tensor
```

HBM 相对 CPU 内存带宽很高，但相对 SM 内部的计算与片上存储仍然很“远”。因此，GPU 性能分析不能只看计算量，也要看数据搬运。

## 2. Register：Thread 私有的高速空间

Kernel 中的局部变量通常优先使用 Register，例如：

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
float a = A[i];
float b = B[i];
float c = a + b;
```

可以先把 `i`、`a`、`b`、`c` 理解为当前 Thread 正在使用的小数据。

```text
Thread 0                    Thread 1
└── Registers               └── Registers
    ├── a                       ├── a
    ├── b                       ├── b
    └── c                       └── c
```

两边的 `a` 不是同一个变量。Register 的核心特点是：

- 非常靠近计算单元，访问很快；
- 基本由单个 Thread 私有；
- 容量有限，不能无限使用。

## 3. Shared Memory：Block 内的共享工作台

Shared Memory 是同一个 Block 内 Threads 可以共同访问的高速片上存储。

```text
Block
├── Thread 0 ─┐
├── Thread 1 ─┼── Shared Memory
├── Thread 2 ─┤
└── ...      ─┘
```

它适合这样的场景：多个 Thread 需要反复使用同一小块数据。Threads 可以协作把数据从 HBM 搬进 Shared Memory，之后从更近的位置重复读取。

### 关键纠正：共享范围是 Block，不是整个 SM

学习者原理解：

> Shared Memory 是 SM 内部多线程可以共享的内存。

更准确的说法是：

> Shared Memory 的协作与可见范围是同一个 Block。即使多个 Block 同时驻留在同一 SM 上，它们也不能把 Shared Memory 当成彼此随意共享的公共空间。

## 4. 三者对比

| 存储 | 主要使用范围 | 容量直觉 | 速度直觉 | 典型用途 |
|---|---|---|---|---|
| HBM | 整个 GPU 上的大数据 | 大 | 相对慢、距离远 | 输入和输出 Tensor |
| Shared Memory | 同一 Block | 小 | 很快 | Block 内协作和数据复用 |
| Register | 单个 Thread | 更小 | 极快 | 局部变量、中间结果 |

这是当前阶段的简化模型，目的是建立性能直觉，而不是描述完整的 GPU Memory Hierarchy。

## 5. Data Reuse：为什么要把数据搬近

把数据从 HBM 搬进 Shared Memory 本身也有成本。只有数据会被多次使用时，这次搬运才可能值得。

```text
HBM 中的一块数据
        ↓ 搬一次
Shared Memory
        ↓ 被多个 Thread 多次使用
减少重复访问 HBM
```

### Tile 是什么

当前阶段只需把 Tile 理解为“大数据中的一小块”。整个矩阵放不进 Shared Memory，所以将其分成小块，一块一块搬入片上存储并计算。

```text
大矩阵 → 切成小块（Tiles）→ 一块一块搬入 Shared Memory → 重复使用
```

理解 Tile 暂时不要求扎实的线性代数基础。

## 6. MatMul 与 Vector Add 的区别

矩阵乘法中，同一个 A Tile 或 B Tile 往往会参与多次乘加，因此具有明显的数据复用机会。

Vector Add 中：

```cpp
C[i] = A[i] + B[i];
```

每个 Thread 通常只读取一次 `A[i]` 和 `B[i]`，计算一次，再写回 `C[i]`。把它们先搬进 Shared Memory 会增加额外操作，却没有明显复用收益。

> Shared Memory 不是“用了就更快”。没有 Data Reuse 时，通常不值得使用。

## 7. Register 使用过多会怎样

学习者原理解：

> 每个 Thread 使用的 Register 太多，会让可执行的线程数减少，反而增加 latency。

前半段正确，但最后需要更精确：

```text
每 Thread 使用太多 Register
        ↓
SM 的 Register 总量有限
        ↓
同时驻留的 Threads / Warps 变少
        ↓
Ready Warps 可能变少
        ↓
隐藏等待时间的能力下降
```

不是 HBM 的访问延迟本身变长，而是 SM 缺少足够的 Ready Warps 去覆盖这段等待。

## 学习者问答整理

### Q1：Tile 不理解，但知道 MatMul 可以复用，Add 不可以，这样理解对吗？

对。现阶段记住 `Tile = 大数据中的一小块` 即可。MatMul 的 Tile 会反复参与计算；Vector Add 的元素通常只使用一次，所以 Shared Memory 很难带来收益。

### Q2：Shared Memory 与 Register 的区别？

- Register：Thread 私有；
- Shared Memory：同一个 Block 内的 Threads 共享。

### Q3：为什么不能把所有数据都放在 Register？

因为 Register 容量有限。每个 Thread 占用过多 Register 会减少 SM 上可同时驻留的 Warps，从而削弱 Latency Hiding 能力。

## 本节最小记忆集

> HBM 大但远；Register 快但由 Thread 私有且容量很小；Shared Memory 是 Block 内的高速共享工作台。只有存在 Data Reuse 时，把数据搬进 Shared Memory 才可能值得。
