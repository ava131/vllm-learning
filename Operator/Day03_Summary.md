# Day 3 总结：Memory Coalescing、二维索引与 Tiled Transpose

## Day 3 学习目标

Day 3 的主线是：

> 同样的数学计算，为什么不同的 Thread → Data Mapping 会让 Kernel 性能差很多。

Day 1 看的是算子整体瓶颈，Day 2 看的是 CUDA 执行层级，Day 3 开始真正进入 GPU Kernel 的性能分析：

```text
Thread 怎么映射数据
        ↓
Warp 怎么访问 HBM
        ↓
Memory Coalescing 好不好
        ↓
Data Reuse 高不高
        ↓
Shared Memory 有没有意义
        ↓
Kernel 更可能 Memory-bound 还是 Compute-bound
```

## 1. Memory Coalescing

Memory Coalescing 关注的是：

> 一个 Warp 里的 32 个 Threads 访问 HBM 时，地址是不是连续、规整，从而能用更少的 memory transactions 搬到更多有用数据。

例如：

```text
T0  → A[256]
T1  → A[257]
T2  → A[258]
...
T31 → A[287]
```

这是典型 coalesced access。

因为 32 个 Threads 访问的是连续 float：

```text
32 floats × 4 Bytes = 128 Bytes
```

硬件可以用很少的 memory transactions 把这一段连续数据搬进来。

反过来：

```text
T0  → A[0]
T1  → A[32]
T2  → A[64]
...
T31 → A[992]
```

这是 strided / scattered access。

虽然每个 Thread 还是只读 1 个 float，但 Warp 整体访问的地址跨度很大，可能需要更多 memory transactions。每次 transaction 里真正被用到的 useful bytes 变少，effective bandwidth 就会下降。

## 2. Warp-Level Memory Access

分析 coalescing 时，重点不是看单个 Thread，也不是看整个 Block，而是看一个 Warp。

```text
Block
└── Warps
    └── Threads
```

原因是：

- Kernel 代码从编程语义上是每个 Thread 执行；
- 但硬件调度和执行时，Warp 是非常重要的粒度；
- NVIDIA GPU 中通常 1 Warp = 32 Threads；
- 一个 Warp 内 Threads 如果访问连续地址，HBM 访问效率通常更好。

所以分析内存访问时，要问：

```text
同一个 Warp 内的 32 个 Threads
访问的是连续地址吗？
还是 stride 很大？
```

## 3. Data Reuse vs Coalescing

Data Reuse 和 Coalescing 不是一个东西。

| 概念 | 关注点 | 典型问题 |
|---|---|---|
| Coalescing | Warp 内 Threads 的 HBM 地址是否连续 | 这 32 个 Threads 访问是不是连着的？ |
| Data Reuse | 一个数据被读进来后是否会被多次使用 | 搬进来的数据会不会被反复用？ |

一个 Kernel 完全可能：

```text
Coalescing = 好
Data Reuse = 低
```

例如 Vector Add：

```cpp
C[i] = A[i] + B[i];
```

如果 `i` 连续，那么 Warp 访问 A、B、C 都是 coalesced。

但 `A[i]` 和 `B[i]` 基本只用一次：

```text
从 HBM 读进来
        ↓
加一次
        ↓
没用了
```

所以 Data Reuse 很低，Shared Memory 通常没什么收益。

## 4. Row-Major 与二维索引

C/C++ 中二维数组按 row-major 存储：

```text
matrix[row][col]
```

展开成一维时：

```text
linear_index = row * num_columns + col
```

CUDA 里常用：

```text
x → 列方向
y → 行方向
```

所以访问二维矩阵 A 时通常写：

```cpp
A[y * width + x]
```

这里：

```text
y      = row
x      = col
width  = A 的列数
```

最小记忆：

```text
A[y][x]
= A[row][col]
= A[y * width + x]
```

容易混的点是：数组索引顺序是 `[row][col]`，但 CUDA 坐标命名通常是 `(x, y)`。因此不是 `A[x][y]`，而是 `A[y][x]`。

## 5. 二维 Grid / Block / Thread

一维 Vector Add 里常见：

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

二维矩阵里，只是把这个思想扩展成 x 和 y 两个方向：

```cpp
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
```

含义：

```text
blockIdx.x / blockIdx.y
        ↓
当前 Block 在 Grid 里的二维位置

threadIdx.x / threadIdx.y
        ↓
当前 Thread 在 Block 里的二维位置

x / y
        ↓
当前 Thread 对应到全局矩阵中的列 / 行
```

例如：

```text
blockDim.x  = 32
blockDim.y  = 32
blockIdx.x  = 2
blockIdx.y  = 3
threadIdx.x = 5
threadIdx.y = 7
```

那么：

```text
x = 2 × 32 + 5 = 69
y = 3 × 32 + 7 = 103
```

这个 Thread 访问：

```text
A[y][x] = A[103][69]
```

## 6. Naive Transpose 为什么慢

矩阵转置：

```text
B = A^T
```

也就是：

```text
B[x][y] = A[y][x]
```

如果 A 是 row-major，那么按行读 A 通常容易 coalesced：

```text
T0  → A[y][x + 0]
T1  → A[y][x + 1]
T2  → A[y][x + 2]
...
```

但是 naive transpose 直接写 B 时，很容易变成按列写：

```text
T0  → B[x + 0][y]
T1  → B[x + 1][y]
T2  → B[x + 2][y]
...
```

在 row-major 中，连续的内存方向是列方向，也就是同一行内 `col` 连续。按列写通常 stride 很大。

所以 naive transpose 常见问题是：

```text
读 A：coalesced
写 B：strided / not coalesced
```

结果是 memory transactions 变多，effective bandwidth 下降。

## 7. Tiled Transpose 的核心想法

Tiled transpose 使用 Shared Memory 做中转：

```text
HBM A
  ↓
Warp 按行连续读 A
  ↓
Shared Memory tile
  ↓
在 tile 内交换行列
  ↓
Warp 按行连续写 B
  ↓
HBM B
```

核心不是“Shared Memory 比 HBM 快，所以随便用都快”。

更准确是：

> Shared Memory 让我们把 HBM 中昂贵的 strided access，变成片上内存里的 data rearrangement，从而尽量让 HBM 读和 HBM 写都 coalesced。

Shared Memory 在这里有两个作用：

1. **Data Reuse**
   数据如果会被同一个 Block 内多个 Threads 多次使用，Shared Memory 可以减少反复从 HBM 读取。

2. **Data Rearrangement**
   即使没有很强的 reuse，也可以先把数据按 coalesced 方式读进 tile，再在 Shared Memory 中换一种布局读出，让 HBM 写入也保持 coalesced。

Transpose 主要体现的是第二点：data rearrangement。

## 8. Tiled Transpose Kernel

教学版代码：

```cpp
#define TILE 32

__global__ void transpose_tiled(
    const float* A,
    float* B,
    int width,
    int height
) {
    __shared__ float tile[TILE][TILE];

    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;

    if (x < width && y < height) {
        tile[threadIdx.y][threadIdx.x] = A[y * width + x];
    }

    __syncthreads();

    int out_x = blockIdx.y * TILE + threadIdx.x;
    int out_y = blockIdx.x * TILE + threadIdx.y;

    if (out_x < height && out_y < width) {
        B[out_y * height + out_x] = tile[threadIdx.x][threadIdx.y];
    }
}
```

## 9. `__syncthreads()` 的作用

`__syncthreads()` 是 Block 内同步。

在 tiled transpose 中，它保证：

```text
所有 Threads 都已经把 A 的数据写入 tile
        ↓
之后才允许任何 Thread 从 tile 读取数据写 B
```

如果没有同步，可能出现：

```text
某些 Threads 还没写完 tile
        ↓
另一些 Threads 已经开始读 tile
        ↓
读到旧值 / 未定义值
```

注意：

- `__syncthreads()` 只同步同一个 Block 内的 Threads；
- 不同步不同 Blocks；
- 所以 Shared Memory 也只在同一个 Block 内共享。

## 10. A height × width，B width × height

如果输入矩阵：

```text
A shape = height × width
```

那么转置后：

```text
B shape = width × height
```

也就是说：

```text
A 的行数 = height
A 的列数 = width

B 的行数 = width
B 的列数 = height
```

row-major 展开公式永远是：

```text
index = row * 当前矩阵的列数 + col
```

所以 A：

```cpp
A[y * width + x]
```

因为 A 每一行有 `width` 个元素。

而 B：

```cpp
B[out_y * height + out_x]
```

因为 B 每一行有 `height` 个元素。

这不是写错，也不是随便换变量名，而是矩阵 shape 变了。

非方阵最清楚：

```text
A: 2 × 3

a b c
d e f

B = A^T: 3 × 2

a d
b e
c f
```

A 的列数是 3，所以 A 用 `width = 3`。

B 的列数是 2，而这个 2 正好是原来 A 的 `height`。

因此：

```text
A[row][col] → A[row * width + col]
B[row][col] → B[row * height + col]
```

## 11. Block(x,y) → Block(y,x)

Transpose 可以拆成两层：

```text
大矩阵层面：
Block(x, y) → Block(y, x)

Tile 内部：
tile[ty][tx] → tile[tx][ty]
```

输入时：

```cpp
int x = blockIdx.x * TILE + threadIdx.x;
int y = blockIdx.y * TILE + threadIdx.y;
```

当前 Block 负责 A 中的：

```text
Block(blockIdx.x, blockIdx.y)
```

输出时：

```cpp
int out_x = blockIdx.y * TILE + threadIdx.x;
int out_y = blockIdx.x * TILE + threadIdx.y;
```

这相当于把 Tile 的整体位置交换：

```text
A 中 Block(x, y)
        ↓
B 中 Block(y, x)
```

但是输出地址仍然让 `threadIdx.x` 对应 `out_x`，因为一个 Warp 内通常 `threadIdx.x` 连续变化，这样写 B 时仍然容易 coalesced。

## 12. tile[ty][tx] → tile[tx][ty]

读 A 到 Shared Memory：

```cpp
tile[threadIdx.y][threadIdx.x] = A[y * width + x];
```

也就是：

```text
tile[ty][tx] = A[y][x]
```

写 B 时：

```cpp
B[out_y * height + out_x] = tile[threadIdx.x][threadIdx.y];
```

也就是：

```text
从 tile[tx][ty] 读
```

这一步完成 Tile 内部 transpose。

用 4 × 4 的 tile 看：

```text
tile:

a b c d
e f g h
i j k l
m n o p
```

如果某个简化版 Warp 的 Threads 是同一行：

```text
T0: tx=0, ty=0
T1: tx=1, ty=0
T2: tx=2, ty=0
T3: tx=3, ty=0
```

读取：

```text
tile[tx][ty]
```

得到：

```text
T0 → tile[0][0] → a
T1 → tile[1][0] → e
T2 → tile[2][0] → i
T3 → tile[3][0] → m
```

这正好是转置后的一行：

```text
a e i m
```

所以：

```text
Block 位置交换
        ↓
Tile 被放到 B 中正确的大位置

tile 索引交换
        ↓
Tile 内部数据被转置

threadIdx.x 继续对应 out_x
        ↓
Warp 写 B 时仍然连续
```

## 13. Kernel、Thread、Warp 的关系

需要固定住这两个视角：

### 编程语义

```text
Kernel 代码描述的是每个 Thread 要执行什么。
```

例如：

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
C[i] = A[i] + B[i];
```

每个 Thread 都执行这份代码，只是看到的 `blockIdx` 和 `threadIdx` 不同，所以算出来的 `i` 不同。

### 硬件执行

```text
GPU 会把 Block 内的 Threads 组织成 Warps。
```

通常：

```text
1 Warp = 32 Threads
```

Warp Scheduler 调度 Ready Warps 执行。某个 Warp 等 HBM 时，SM 可以切到其他 Ready Warp，形成 Day 2 讲过的 latency hiding。

因此：

```text
每个 Thread 执行 Kernel

但分析 coalescing / 调度 / latency hiding 时
Warp 是非常重要的粒度
```

这两句话不矛盾。

## 14. P4：Vector Add Kernel 性能分析

Kernel A：

```cpp
__global__ void add_A(
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

启动：

```cpp
int threads = 256;
int blocks = (N + threads - 1) / threads;

add_A<<<blocks, threads>>>(A, B, C, N);
```

### Thread Mapping

如果：

```text
blockIdx.x  = 3
blockDim.x  = 256
threadIdx.x = 7
```

那么：

```text
i = 3 × 256 + 7 = 775
```

这个 Thread 负责：

```text
A[775] + B[775] → C[775]
```

### Coalescing

如果某个 Warp 对应：

```text
i = 256 ~ 287
```

那么：

```text
T0  → A[256]
T1  → A[257]
T2  → A[258]
...
T31 → A[287]
```

B 和 C 也是连续的。

所以：

```text
Read A：coalesced
Read B：coalesced
Write C：coalesced
```

### Data Reuse

Vector Add 中：

```text
A[i] 用一次
B[i] 用一次
C[i] 写一次
```

Data Reuse 很低。

所以 Shared Memory 没有明显意义，因为把数据搬进 Shared Memory 后也不会被反复使用。

### Arithmetic Intensity

每个元素大致：

```text
Read A[i]   → 4 Bytes
Read B[i]   → 4 Bytes
Add         → 1 FLOP
Write C[i]  → 4 Bytes
```

简化估算：

```text
12 Bytes data movement
1 FLOP
```

Arithmetic Intensity：

```text
AI = 1 / 12 ≈ 0.083 FLOP / Byte
```

计算量很少，数据搬运很多，所以 Vector Add 通常是 memory-bound。

## 15. P4：Stride = 32 的坏 Mapping

Kernel B：

```cpp
__global__ void add_B(
    const float* A,
    const float* B,
    float* C,
    int N
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int i = tid * 32;

    if (i < N) {
        C[i] = A[i] + B[i];
    }
}
```

一个 Warp 中：

```text
T0  → tid = 0 → A[0]
T1  → tid = 1 → A[32]
T2  → tid = 2 → A[64]
...
T31 → tid = 31 → A[992]
```

这就是 stride = 32 的访问。

虽然每个被处理的元素仍然是：

```text
1 load A
1 load B
1 add
1 store C
```

但 Warp 视角完全变差了：

```text
地址不连续
        ↓
memory transactions 变多
        ↓
每次 transaction 的 useful bytes 变少
        ↓
effective bandwidth 降低
        ↓
Kernel 可能明显变慢
```

这说明：

> 同一个数学 operator，不同的 Thread Mapping 会导致完全不同的 memory access pattern。

## 16. Day 3 核心心智模型

### 模型一：先看 Mapping，再看计算

```text
不要先盯着 C[i] = A[i] + B[i]

先问：
这个 Thread 的 i / x / y 是怎么算出来的？
```

Mapping 决定：

- 每个 Thread 负责哪个元素；
- 一个 Warp 是否访问连续地址；
- HBM transaction 是否高效；
- Kernel 可能慢在哪里。

### 模型二：Coalescing 看 Warp

```text
单个 Thread 访问没法判断 coalescing。
整个 Block 也不是最直接的判断粒度。

要看同一个 Warp 内 32 个 Threads 的地址模式。
```

### 模型三：Shared Memory 不是默认加速器

Shared Memory 有意义通常因为：

```text
Data Reuse
或
Data Rearrangement
```

如果数据只用一次，Shared Memory 往往帮不上忙，甚至会增加额外搬运和同步成本。

### 模型四：Transpose 拆成两层

```text
Block(x,y) → Block(y,x)
        ↓
Tile 的大位置转置

tile[ty][tx] → tile[tx][ty]
        ↓
Tile 内部数据转置
```

### 模型五：线性索引永远乘“当前矩阵的列数”

```text
index = row * num_columns + col
```

所以：

```text
A shape = height × width
A[y * width + x]

B shape = width × height
B[out_y * height + out_x]
```

## 17. 易错点

### 易错点 1：把 x/y 和 row/col 搞反

CUDA 坐标习惯：

```text
x = col
y = row
```

C/C++ 数组习惯：

```text
A[row][col]
```

合起来就是：

```text
A[y][x]
```

### 易错点 2：以为 B 也要乘 width

不一定。

展开二维数组时乘的是当前矩阵的列数。

Transpose 后：

```text
B shape = width × height
```

所以 B 的列数是 `height`。

因此：

```cpp
B[out_y * height + out_x]
```

### 易错点 3：把 Block 当成一次 memory access 的单位

Coalescing 主要看 Warp。

一个 `32 × 32` Block 有：

```text
1024 Threads = 32 Warps
```

如果 `blockDim.x = 32`，每个 Warp 很直观地对应一行 Threads。但这不是所有二维 Block 都天然成立；它成立是因为 x 方向刚好 32。

### 易错点 4：以为 Thread x/y 也必须在输出时交换

Tiled transpose 中，输出坐标故意保留：

```cpp
out_x = blockIdx.y * TILE + threadIdx.x;
out_y = blockIdx.x * TILE + threadIdx.y;
```

Block 的 x/y 交换了，但 Thread 的 x/y 没有直接交换。

原因是：

```text
threadIdx.x 连续
        ↓
out_x 连续
        ↓
B 的写入 coalesced
```

Tile 内部真正的数据交换交给：

```cpp
tile[threadIdx.x][threadIdx.y]
```

### 易错点 5：以为 Shared Memory 只用于 Data Reuse

Shared Memory 当然常用于 data reuse，但 transpose 里它还用于 data rearrangement。

也就是：

```text
先用 coalesced 方式从 HBM 读
        ↓
在 Shared Memory 里重排
        ↓
再用 coalesced 方式写回 HBM
```

## 18. 面试速记

### Memory Coalescing

> Coalescing means threads in the same warp access contiguous and aligned memory locations, allowing the GPU to combine their memory requests into fewer transactions and achieve higher effective bandwidth.

### Data Reuse vs Coalescing

> Coalescing is about the memory access pattern across threads in a warp. Data reuse is about whether loaded data is used multiple times. A kernel can have good coalescing but low data reuse, such as vector add.

### Vector Add Bottleneck

> Vector add has low arithmetic intensity: about one FLOP for roughly 12 bytes of memory traffic in a simple model. Therefore it is typically memory-bound, not compute-bound.

### Strided Access

> With stride-32 access, each thread in a warp touches memory far apart from its neighbors. This increases memory transactions, reduces useful bytes per transaction, lowers effective bandwidth, and can make the kernel much slower.

### Tiled Transpose

> Naive transpose usually has coalesced reads but strided writes. Tiled transpose uses shared memory as a staging area: read A coalesced into tile[ty][tx], synchronize, then write B coalesced by reading tile[tx][ty].

### Kernel / Thread / Warp

> A CUDA kernel describes what each logical thread executes. The hardware groups threads into warps, typically 32 threads per warp, which are the important execution and scheduling units.

## 19. Day 3 完成状态

Day 3 已完成：

- 理解 Memory Coalescing；
- 能从 Warp 视角判断连续访问和 stride 访问；
- 能区分 Data Reuse 和 Coalescing；
- 理解 Shared Memory 不只是“更快的内存”，还可以承担 data rearrangement；
- 理解 row-major 下 `A[y * width + x]` 的来源；
- 理解二维 `blockIdx` / `threadIdx` 如何映射到矩阵坐标；
- 理解 naive transpose 为什么写 B 容易 stride；
- 理解 tiled transpose 中 `Block(x,y) → Block(y,x)`；
- 理解 tiled transpose 中 `tile[ty][tx] → tile[tx][ty]`；
- 理解 `__syncthreads()` 是 Block 内同步；
- 理解 A 是 `height × width`，B 是 `width × height`，所以 B 线性索引用 `height`；
- 能独立分析 Vector Add 的 thread mapping、coalescing、data reuse、arithmetic intensity 和 memory-bound；
- 能解释 stride = 32 为什么导致更多 memory transactions、useful bytes 下降、effective bandwidth 下降。

## 20. Day 3 最小记忆集

> Kernel 是每个 Thread 执行的代码；Warp 是硬件调度和执行的重要粒度。分析 Memory Coalescing 时看一个 Warp 内 32 个 Threads 的地址是否连续。Data Reuse 和 Coalescing 不是一回事：Vector Add 可以 coalesced 很好，但 data reuse 很低，因此通常不需要 Shared Memory，且因为 AI 很低而 memory-bound。二维矩阵中 x 是列、y 是行，所以 row-major 访问是 `A[y * width + x]`。Transpose 后 B 的 shape 是 `width × height`，所以 B 的线性索引是 `B[out_y * height + out_x]`。Tiled Transpose 的核心是：`Block(x,y) → Block(y,x)` 负责 tile 的大位置转置，`tile[ty][tx] → tile[tx][ty]` 负责 tile 内部转置，Shared Memory 让 HBM 读写尽量都保持 coalesced。

