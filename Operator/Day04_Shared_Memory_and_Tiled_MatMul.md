# Operator Learning 2026 - Day 04

# Shared Memory Bank Conflict and Tiled MatMul

本日主题：

```text
Shared Memory Bank Conflict
MatMul Tiling / Data Reuse
基础 Tiled MatMul Kernel
Naive vs Tiled MatMul 性能分析
```

Day4 的核心不是“会背 Shared Memory 很快”，而是要建立一条完整的数据流：

```text
HBM
 │
 │ coalesced global load
 ▼
Shared Memory
 │
 ├─ Bank Conflict?
 └─ Block 内 Data Reuse
 ▼
Registers
 │
 │ thread-private partial sum
 ▼
Compute
```

同时也要建立执行层心智模型：

```text
Grid
 ↓
Block → 负责一个 C Tile
 ↓
Threads cooperative load
 ↓
每个 Thread → 一个或多个 C elements
 ↓
Register 保存 partial sum
 ↓
最后写回 C
```

---

## 1. Shared Memory 是什么

Shared Memory 是一个 CUDA Block 内所有 Threads 可以共享的片上高速内存。

它常见用途不是“替代 HBM”，而是：

```text
HBM 中的数据
↓
由 Block 内 Threads 合作搬进 Shared Memory
↓
Block 内多个 Threads 多次使用
↓
减少重复 HBM 访问
```

一个关键边界：

> Shared Memory 只在同一个 CUDA Block 内共享。不同 Block 之间不能直接共享同一块 Shared Memory。

所以：

```text
Block 0 的 Shared Memory
只给 Block 0 的 Threads 用

Block 1 的 Shared Memory
只给 Block 1 的 Threads 用
```

这会影响 MatMul 的 reuse 分析：Shared Memory 能显式组织的是 **Block 内 reuse**，不是跨 Block reuse。

---

## 2. Bank 是什么

Shared Memory 虽然快，但它内部不是一个“无限并行”的黑盒。

经典简化模型：

```text
Shared Memory
↓
32 Banks
```

一个 Warp 也是：

```text
32 Threads
```

理想情况：

```text
T0  → Bank 0
T1  → Bank 1
T2  → Bank 2
...
T31 → Bank 31
```

这样 32 个线程可以更好地并行访问 Shared Memory。

可以暂时把 Bank 理解成：

> Shared Memory 内部的并行访问通道 / 分区资源。

但不要把 Bank 理解成：

```text
Bank = 一行
Bank = 一列
Bank = 一个线程
```

Bank 本质上是地址映射到的硬件通道。

---

## 3. float 地址如何映射到 Bank

在经典简化模型下：

```text
float = 4 Bytes
32 Banks
连续 32-bit word 轮流进入 Bank
```

也就是：

```text
element 0  → Bank 0
element 1  → Bank 1
element 2  → Bank 2
...
element 31 → Bank 31
element 32 → Bank 0
element 33 → Bank 1
...
```

近似公式：

```text
bank = element_index % 32
```

换成 byte address，可以理解为：

```text
0~3      → Bank 0
4~7      → Bank 1
8~11     → Bank 2
...
124~127  → Bank 31
128~131  → Bank 0
```

所以更准确的说法是：

```text
每连续 4 Bytes 换下一个 Bank
每 32 个 float，也就是 128 Bytes，Bank 编号循环一次
```

不是“每 32 Bytes 一个 Bank”。

---

## 4. Bank Conflict

Bank Conflict 关注的是：

> 一个 Warp 内多个 Threads 访问 Shared Memory 时，是否争抢同一个 Bank。

如果一个 Warp 中 32 个 Threads 分散到 32 个 Bank：

```text
T0  → Bank0
T1  → Bank1
...
T31 → Bank31
```

访问效率好。

如果它们都访问 Bank0 的不同地址：

```text
T0  ─┐
T1  ─┤
T2  ─┤
...  ├→ Bank0
T31 ─┘
```

就会发生严重 Bank Conflict。

注意：

```text
Bank Conflict 是性能问题，不是 correctness 问题。
```

程序不一定算错，但 Shared Memory 的并行访问能力没有发挥出来。

真实硬件还有 broadcast / multicast、访问宽度、架构差异等细节。当前阶段只需要记住：

> 一个 Warp 内多个 Threads 如果访问同一个 Bank 的不同地址，就可能发生 Bank Conflict，降低 Shared Memory 访问效率。

---

## 5. 为什么 tile[32][32] 转置读取会冲突

经典 transpose 中常见：

```cpp
__shared__ float tile[32][32];
```

Shared Memory 中的二维数组仍然是 row-major。

所以：

```text
tile[0][0] → element 0
tile[0][1] → element 1
...
tile[0][31] → element 31
tile[1][0] → element 32
tile[1][1] → element 33
...
```

按行访问：

```text
T0  → tile[0][0] → element 0  → Bank 0
T1  → tile[0][1] → element 1  → Bank 1
...
T31 → tile[0][31] → element 31 → Bank 31
```

很好，没有冲突。

但 transpose 阶段经常要按列读：

```text
T0  → tile[0][0]
T1  → tile[1][0]
T2  → tile[2][0]
...
T31 → tile[31][0]
```

对应 element index：

```text
0
32
64
96
...
31 × 32
```

算 Bank：

```text
0  % 32 = 0
32 % 32 = 0
64 % 32 = 0
96 % 32 = 0
...
```

所以整个 Warp 都撞到 Bank0。

这就是：

```text
tile[32][32]
↓
列访问 stride = 32 floats
↓
stride % 32 = 0
↓
同一列落到同一个 Bank
↓
严重 Bank Conflict
```

---

## 6. 为什么 tile[32][33] 能解决

经典修正：

```cpp
__shared__ float tile[32][33];
```

逻辑 Tile 还是 `32×32`，多出来的 1 列只是 padding。

它的作用不是“多开一点内存所以更快”，而是：

> 故意改变 Shared Memory 中每一行的 stride。

原来：

```text
stride = 32 floats
```

现在：

```text
stride = 33 floats
```

按列读第一列：

```text
tile[0][0] → element 0
tile[1][0] → element 33
tile[2][0] → element 66
tile[3][0] → element 99
...
```

算 Bank：

```text
0  % 32 = 0
33 % 32 = 1
66 % 32 = 2
99 % 32 = 3
...
```

于是：

```text
T0  → Bank 0
T1  → Bank 1
T2  → Bank 2
...
T31 → Bank 31
```

Bank 被打散了。

---

## 7. 4x4 / 4 Banks 玩具模型

先用一个小模型看清楚：

```cpp
__shared__ float tile[4][4];
```

假设只有 4 个 Banks。

Row-major 下：

```text
element 0 → Bank0
element 1 → Bank1
element 2 → Bank2
element 3 → Bank3
element 4 → Bank0
...
```

所以二维上看：

```text
tile[4][4]:

row0: B0 B1 B2 B3
row1: B0 B1 B2 B3
row2: B0 B1 B2 B3
row3: B0 B1 B2 B3
```

按列读：

```text
T0 → tile[0][0] → Bank0
T1 → tile[1][0] → Bank0
T2 → tile[2][0] → Bank0
T3 → tile[3][0] → Bank0
```

冲突。

加 padding：

```cpp
__shared__ float tile[4][5];
```

此时每行 stride = 5。

Bank 图变成：

```text
tile[4][5]:

row0: B0 B1 B2 B3 X
row1: B1 B2 B3 B0 X
row2: B2 B3 B0 B1 X
row3: B3 B0 B1 B2 X
```

再按列读：

```text
T0 → tile[0][0] → Bank0
T1 → tile[1][0] → Bank1
T2 → tile[2][0] → Bank2
T3 → tile[3][0] → Bank3
```

不再挤同一个 Bank。

这就是 `tile[32][33]` 的本质。

---

## 8. Coalescing vs Bank Conflict

这两个概念很容易混。

### Memory Coalescing

发生在：

```text
HBM / global memory
```

关注：

```text
一个 Warp 的 global memory 地址是否连续、是否能被合并成较少 memory transactions
```

目标：

```text
提高 effective bandwidth
减少浪费的 memory transaction
```

### Bank Conflict

发生在：

```text
Shared Memory
```

关注：

```text
一个 Warp 的 Threads 访问 Shared Memory 时是否争抢同一个 Bank
```

目标：

```text
让 Shared Memory 的多个 Banks 并行工作
```

一句话区分：

```text
HBM：看 Warp 的地址能不能高效合并 → Coalescing
Shared Memory：看 Warp 的线程会不会争抢同一 Bank → Bank Conflict
```

---

## 9. MatMul 为什么适合 Tiling

矩阵乘：

```text
A: M × K
B: K × N
C: M × N
```

公式：

```text
C[i][j] = sum_k A[i][k] * B[k][j]
```

一个 `C[i][j]` 需要：

```text
A 的第 i 行
B 的第 j 列
```

Naive CUDA 思路通常是：

```text
一个 Thread 负责一个 C element
```

代码：

```cpp
__global__ void matmul_naive(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;

        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k]
                 * B[k * N + col];
        }

        C[row * N + col] = sum;
    }
}
```

问题是：A/B 中很多元素会被多个输出重复使用。

例如同一行输出：

```text
T0 → C[0][0]
T1 → C[0][1]
T2 → C[0][2]
T3 → C[0][3]
```

当 `k=0`：

```text
T0: A[0][0] * B[0][0]
T1: A[0][0] * B[0][1]
T2: A[0][0] * B[0][2]
T3: A[0][0] * B[0][3]
```

同一个 `A[0][0]` 被多个 Threads 用。

同一列输出：

```text
T0 → C[0][0]
T1 → C[1][0]
T2 → C[2][0]
T3 → C[3][0]
```

当 `k=0`：

```text
T0: A[0][0] * B[0][0]
T1: A[1][0] * B[0][0]
T2: A[2][0] * B[0][0]
T3: A[3][0] * B[0][0]
```

同一个 `B[0][0]` 被多个 Threads 用。

所以 MatMul 天然有大量：

```text
跨 Threads 的 data reuse
```

Shared Memory 的价值就是把这种 reuse 显式组织起来：

```text
HBM load once
↓
Shared Memory
↓
Block 内多个 Threads reuse
```

---

## 10. 为什么 elementwise add 不适合硬塞 Shared Memory

例如：

```cpp
C[i] = A[i] + B[i];
```

一个元素：

```text
A[i] 从 HBM 读进来
↓
用一次
↓
基本没了
```

B 也一样。

如果硬塞 Shared Memory：

```text
HBM → Shared Memory → Thread
```

反而多了一层搬运和同步，但没有 reuse。

所以：

> Shared Memory 不是加速按钮。要问它是为了 reuse，还是为了 rearrange access pattern。

MatMul 适合 Shared Memory，是因为 A/B 的元素会被重复使用。

---

## 11. 8x8 矩阵 + 2x2 Tile 示例

假设：

```text
A: 8 × 8
B: 8 × 8
C: 8 × 8

Tile = 2 × 2
```

C 会被切成：

```text
4 × 4 = 16 个 C Tiles
```

概念上：

```text
16 个 CUDA Blocks
↓
每个 Block 负责一个 2×2 C Tile
```

每个 Block 内：

```text
4 Threads
↓
每个 Thread 负责一个 C element
↓
每个 Thread 用 Register 保存自己的 sum
```

例如 Block(0,0) 负责：

```text
C00 C01
C10 C11
```

四个线程：

```text
T00 → C00
T01 → C01
T10 → C10
T11 → C11
```

因为：

```text
K = 8
Tile K = 2
```

所以一个 C Tile 要经过 4 轮 K Tile：

```text
Round 0: k = 0,1
Round 1: k = 2,3
Round 2: k = 4,5
Round 3: k = 6,7
```

每一轮：

```text
Load A Tile + B Tile
↓
Shared Memory
↓
Compute partial sum
↓
进入下一轮
```

---

## 12. 2x2 Tile 第一轮具体怎么复用

第一轮搬入：

```text
As:

A00 A01
A10 A11

Bs:

B00 B01
B10 B11
```

四个输出的 partial sum：

```text
C00 partial =
A00 * B00 +
A01 * B10

C01 partial =
A00 * B01 +
A01 * B11

C10 partial =
A10 * B00 +
A11 * B10

C11 partial =
A10 * B01 +
A11 * B11
```

这里 reuse 非常直观：

```text
A00 被 C00、C01 使用
A01 被 C00、C01 使用

B00 被 C00、C10 使用
B01 被 C01、C11 使用
```

也就是：

```text
一个 A 元素 → 被同一输出行的多个列线程使用
一个 B 元素 → 被同一输出列的多个行线程使用
```

放大到 `32×32 Tile`：

```text
一个 A Tile 元素大约被 32 个不同列线程使用
一个 B Tile 元素大约被 32 个不同行线程使用
```

---

## 13. Block / Tile / Thread 的关系

这部分是 Day4 最重要的心智模型之一。

```text
CUDA Block = 工人团队
C Tile = 这个团队要产出的区域
A Tile / B Tile = 这个团队计算时需要搬进来的原材料
Shared Memory = 团队共享的工作台
Thread = 团队里的单个工人
Register = 每个工人手里的临时记事本
```

更技术化地说：

```text
一个 CUDA Block
↓
负责一个 C Tile
↓
Block 内多个 Threads 合作搬 A/B Tile
↓
这些 Threads 重复使用 Shared Memory 中的 A/B Tile
↓
每个 Thread 负责自己的 C element
↓
每个 Thread 用 register 保存 sum
```

需要特别记住：

> Kernel 代码看起来是“一个 Thread 执行的代码”，但理解整体行为时，要把 Block 内所有 Threads 同时展开。

例如代码只有：

```cpp
As[threadIdx.y][threadIdx.x] = ...;
```

不要误解成：

```text
As 只写了一个元素
```

而要展开成：

```text
T00 写 As[0][0]
T01 写 As[0][1]
T10 写 As[1][0]
T11 写 As[1][1]
...
```

所以一个完整 Tile 是由多个 Threads 合作搬出来的。

---

## 14. 基础 Tiled MatMul Kernel

下面是教学版 `2×2 Tile` Kernel。它不是工业级 GEMM，只用于理解 Tiling。

```cpp
__global__ void matmul_tiled(
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K
) {
    __shared__ float As[2][2];
    __shared__ float Bs[2][2];

    int row = blockIdx.y * 2 + threadIdx.y;
    int col = blockIdx.x * 2 + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < K / 2; ++t) {

        As[threadIdx.y][threadIdx.x]
            = A[row * K + (t * 2 + threadIdx.x)];

        Bs[threadIdx.y][threadIdx.x]
            = B[(t * 2 + threadIdx.y) * N + col];

        __syncthreads();

        for (int k = 0; k < 2; ++k) {
            sum += As[threadIdx.y][k]
                 * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    C[row * N + col] = sum;
}
```

这里的简化假设：

```text
Tile size = 2
blockDim = (2, 2)
K 能被 2 整除
暂时没有处理边界
一个 Thread 算一个 C element
```

真实 Kernel 需要处理边界、更多 tile size、register tiling、vectorized load、Tensor Core 等，这些不是 Day4 的目标。

---

## 15. Kernel 中每段代码的意义

### 15.1 Block 负责哪个 C Tile

```cpp
int row = blockIdx.y * 2 + threadIdx.y;
int col = blockIdx.x * 2 + threadIdx.x;
```

例如：

```text
blockIdx = (0,0)
```

四个线程：

```text
thread(0,0) → row=0, col=0 → C00
thread(1,0) → row=0, col=1 → C01
thread(0,1) → row=1, col=0 → C10
thread(1,1) → row=1, col=1 → C11
```

所以：

```text
Block(0,0)
↓
负责 C 左上角 2×2 Tile
```

如果：

```text
blockIdx = (1,0)
```

则负责第一行第二个 C Tile：

```text
C[0][2]  C[0][3]
C[1][2]  C[1][3]
```

### 15.2 t 是 K 方向第几个 Tile

```cpp
for (int t = 0; t < K / 2; ++t)
```

这里 `t` 不是 tile 的总个数，而是：

> 当前正在处理 K 方向的第几个 tile / round。

例如：

```text
K = 8
Tile K = 2

t=0 → k 从 0 开始
t=1 → k 从 2 开始
t=2 → k 从 4 开始
t=3 → k 从 6 开始
```

### 15.3 A Tile 怎么加载

```cpp
As[threadIdx.y][threadIdx.x]
    = A[row * K + (t * 2 + threadIdx.x)];
```

以 `Block(0,0), t=0` 为例：

```text
T00 → A[0][0] → As[0][0]
T01 → A[0][1] → As[0][1]
T10 → A[1][0] → As[1][0]
T11 → A[1][1] → As[1][1]
```

所以：

```text
As:

A00 A01
A10 A11
```

### 15.4 B Tile 怎么加载

```cpp
Bs[threadIdx.y][threadIdx.x]
    = B[(t * 2 + threadIdx.y) * N + col];
```

以 `Block(0,0), t=0` 为例：

```text
T00 → B[0][0] → Bs[0][0]
T01 → B[0][1] → Bs[0][1]
T10 → B[1][0] → Bs[1][0]
T11 → B[1][1] → Bs[1][1]
```

所以：

```text
Bs:

B00 B01
B10 B11
```

### 15.5 计算为什么是 As[ty][k] * Bs[k][tx]

```cpp
for (int k = 0; k < 2; ++k) {
    sum += As[threadIdx.y][k]
         * Bs[k][threadIdx.x];
}
```

矩阵乘本质是：

```text
C[row][col]
=
A 的第 row 行
×
B 的第 col 列
```

所以：

```text
As[ty][k] → 固定自己的输出行
Bs[k][tx] → 固定自己的输出列
k         → 沿 K 方向扫
```

例如 T00：

```text
ty=0
tx=0

sum += As[0][k] * Bs[k][0]
```

k=0：

```text
As[0][0] * Bs[0][0]
= A00 * B00
```

k=1：

```text
As[0][1] * Bs[1][0]
= A01 * B10
```

所以：

```text
T00 sum += A00*B00 + A01*B10
```

这是 C00 在当前 K Tile 上的 partial sum。

一个易错点：

```cpp
As[ty][tx] * Bs[ty][tx]
```

是不对的。那样每个 k 循环都会用错位置，无法表达：

```text
A 的 row 行
乘
B 的 col 列
```

---

## 16. 两个 __syncthreads() 的作用

基础 Tiled MatMul 一轮通常是：

```text
Load A/B Tile
↓
__syncthreads()
↓
Compute using Tile
↓
__syncthreads()
↓
Load Next Tile
```

两个同步点保护的是不同的依赖。

### 第一个 __syncthreads()

位置：

```cpp
As[ty][tx] = ...;
Bs[ty][tx] = ...;

__syncthreads();
```

作用：

```text
WRITE → READ
```

防止：

```text
有的 Thread 还没把 Tile 搬完
其他 Thread 已经开始读 Shared Memory 计算
```

因为每个 Thread 计算时会读别人搬进来的数据。

例如 T00 算 C00 需要：

```text
As[0][0]
As[0][1]
Bs[0][0]
Bs[1][0]
```

其中有些元素不是 T00 自己搬的。

所以必须等整个 Block 都搬完。

### 第二个 __syncthreads()

位置：

```cpp
compute using As/Bs

__syncthreads();

load next As/Bs
```

作用：

```text
READ → NEXT WRITE
```

防止：

```text
有的 Thread 还在读当前 round 的 As/Bs
其他更快的 Thread 已经进入下一轮并覆盖 As/Bs
```

如果没有第二个同步，可能出现：

```text
T00 还在读 Round0 的 Shared Memory
T11 已经开始写 Round1 的新数据
```

结果 T00 可能读到一半旧数据、一半新数据，计算错误。

两个同步压缩记法：

```text
第一个：
大家都搬完了吗？
→ 搬完才能开始用

第二个：
大家都用完了吗？
→ 用完才能覆盖 Shared Memory 装下一块 Tile
```

---

## 17. Block 间不能共享 Shared Memory

Tiled MatMul 中，不同 C Tiles 可能会需要相同的 A Tile 或 B Tile。

例如：

```text
Block(0,0) 算 C 左上角 Tile
Block(1,0) 算 C 第一行第二个 Tile
```

它们可能都需要 A 的同一块数据。

但：

```text
Block(0,0) 的 Shared Memory
不能被 Block(1,0) 直接读取
```

所以同一个 A Tile 可能会被不同 Blocks 分别从 HBM 加载到各自的 Shared Memory。

这听起来浪费，但这是 CUDA Block 隔离模型的一部分。

Shared Memory 的 reuse 是：

```text
Block 内多个 Threads 的显式 reuse
```

不是：

```text
所有 Blocks 之间共享同一份 Tile
```

---

## 18. Cache 与跨 Block reuse 的区别

那跨 Block 的相同数据是不是完全没机会复用？

不是。

可能有 cache 帮忙：

```text
HBM
↓
L2 cache / L1 cache
↓
不同 Blocks 之后再读，可能命中 cache
```

但这和 Shared Memory reuse 不一样。

### Shared Memory reuse

特点：

```text
程序显式组织
Block 内可控
需要 __shared__ 和 __syncthreads()
生命周期属于一个 Block
```

### Cache reuse

特点：

```text
硬件自动管理
可能命中，也可能不命中
跨 Block 可能受益
程序员控制更弱
```

所以更准确的表述是：

> Naive MatMul 不是完全没有 reuse，cache 可能帮忙；但程序本身没有显式组织跨 Threads 的 reuse。Tiled MatMul 用 Shared Memory 明确把 Block 内 reuse 抓住。

---

## 19. Naive vs Tiled 的 Arithmetic Intensity

Arithmetic Intensity：

```text
AI = FLOPs / Bytes
```

### Naive MatMul

一个 Thread 算一个 `C[i][j]`：

```cpp
for (int k = 0; k < K; ++k) {
    sum += A[i*K + k] * B[k*N + j];
}
```

每一轮理想化地读：

```text
A[i][k] → 4B
B[k][j] → 4B
```

做：

```text
1 multiply + 1 add ≈ 2 FLOPs
```

忽略写 C 和 cache：

```text
AI ≈ 2 FLOPs / 8 Bytes
   = 0.25 FLOP/Byte
```

这很低。

含义：

```text
每从 HBM 搬 8B，只做约 2 FLOPs
```

### Tiled MatMul，Tile = 32

一个 Block 算一个：

```text
32 × 32 C Tile
```

某一轮 K Tile 加载：

```text
A Tile: 32 × 32 float
B Tile: 32 × 32 float
```

HBM 读取：

```text
32 × 32 × 4 × 2
= 8192 Bytes
= 8 KB
```

这一轮计算：

```text
C Tile 有 32 × 32 = 1024 个输出元素
每个输出做 32 次 multiply + add
```

FLOPs：

```text
1024 × 32 × 2
= 65536 FLOPs
```

所以：

```text
AI = 65536 / 8192
   = 8 FLOP/Byte
```

对比：

```text
Naive:
≈ 0.25 FLOP/Byte

Tiled, Tile=32:
≈ 8 FLOP/Byte
```

约提高：

```text
32x
```

这不是巧合。

原因是：

```text
Tile=32 时
一个 A 元素大约被 32 个不同列线程使用
一个 B 元素大约被 32 个不同行线程使用
```

所以：

```text
HBM load
↓
Shared Memory
↓
reuse ~32 times
```

非常粗略地：

```text
AI ∝ Tile Size
```

---

## 20. 为什么 Tile 不是越大越好

更大的 Tile 通常意味着更高 reuse：

```text
Tile 16 → reuse ~16
Tile 32 → reuse ~32
Tile 64 → reuse ~64
```

但不能无限增大。

### 20.1 Shared Memory 压力

如果 Tile size = T：

```text
A Tile = T × T
B Tile = T × T
```

float 4B，所以 Shared Memory 需求：

```text
2 × T × T × 4 Bytes
```

例子：

```text
T=16
→ 2 × 16 × 16 × 4
→ 2 KB

T=32
→ 2 × 32 × 32 × 4
→ 8 KB

T=64
→ 2 × 64 × 64 × 4
→ 32 KB
```

Tile 大一倍：

```text
Shared Memory 大约变 4 倍
```

### 20.2 Threads / Block 压力

当前教学 Kernel 是：

```text
一个 Thread → 一个 C element
```

所以：

```text
Tile=16 → 256 threads/block
Tile=32 → 1024 threads/block
Tile=64 → 4096 threads/block
```

`Tile=64` 在这种写法下直接不可能。

真实高性能 GEMM 不会简单地让：

```text
一个 Thread 只算一个 C element
然后无限增大 Tile
```

后续会进入：

```text
一个 Thread 计算多个 C elements
Register Tiling / Thread Tile
```

但 Day4 先不展开。

### 20.3 Occupancy 和 latency hiding

一个 Block 占用资源太多，会导致：

```text
一个 SM 同时驻留的 Blocks/Warps 变少
↓
resident warps 下降
↓
latency hiding 能力变差
```

所以优化不是：

```text
Tile 越大越好
```

而是：

```text
Tile 大 → reuse 更好
Tile 大 → shared memory / register / threads 资源压力更大
```

这是 trade-off。

---

## 21. Tiling 也不自动代表高性能

Shared Memory 本身不是加速按钮。

如果加载 A/B Tile 时，global memory access 很差：

```text
T0 → A[0]
T1 → A[100]
T2 → A[200]
...
```

那么：

```text
HBM → Shared Memory
```

这一段仍然可能产生大量 memory transactions。

所以好的 Tiled MatMul 通常同时要求：

```text
1. Global memory load 尽量 coalesced
2. Shared Memory 内做 data reuse
3. Shared Memory access 尽量避免 bank conflict
4. Register 保存 thread-local partial sums
5. Tile size 不能导致资源压力太大
```

---

## 22. Naive vs Tiled 总表

| 方面 | Naive MatMul | Tiled MatMul |
|---|---|---|
| HBM 访问 | 大量重复读取 A/B | Tile 搬入后 Block 内重复利用 |
| Data Reuse | 程序没有显式组织 | 显式组织 Block 内 reuse |
| Shared Memory | 基本不用 | 核心中转站 |
| Arithmetic Intensity | 较低 | 明显提高 |
| Kernel 复杂度 | 简单 | 更复杂 |
| 同步 | 通常不需要 Block sync | 需要 `__syncthreads()` |
| 资源压力 | 较小 | Shared Memory / Register / Threads 更高 |
| 性能潜力 | 较低 | 高很多 |

但不要形成错觉：

```text
Naive MatMul 一定 memory-bound
Tiled MatMul 一定 compute-bound
```

不一定。

最终还取决于：

```text
shape
dtype
hardware
tile size
cache
Tensor Core usage
memory bandwidth
compute peak
```

更准确的说法：

> Tiling 提高 Arithmetic Intensity，使 MatMul 更有机会从 memory-bound 向 compute-bound 靠近。

---

## 23. Day4 最终 Kernel 性能分析 Checklist

以后看到一个 MatMul Kernel，可以按这个顺序看。

### 1. Thread / Block Mapping

```text
每个 Block 负责什么？
每个 Thread 负责哪个 C element？
一个 Thread 是否只算一个 C，还是算多个 C？
```

### 2. Global Memory Access

```text
一个 Warp 读 A 时是否 coalesced？
一个 Warp 读 B 时是否 coalesced？
写 C 时是否 coalesced？
有没有 strided / scattered access？
memory transaction 的 useful bytes 高不高？
```

### 3. Data Reuse

```text
A/B 元素是否被多个 Threads 重复使用？
reuse 发生在 Block 内，还是依赖 cache？
有没有必要放 Shared Memory？
```

### 4. Shared Memory

```text
Shared Memory 里放了什么？
A Tile / B Tile 怎么组织？
有没有 padding？
Shared Memory access 有没有 Bank Conflict？
同步位置是否正确？
```

### 5. Synchronization

```text
第一个 __syncthreads():
是否保护 load 完整后再 read？

第二个 __syncthreads():
是否保护 read 完成后再覆盖？
```

### 6. Register

```text
partial sum 放在哪里？
每个 Thread 有多少 accumulator？
register pressure 会不会太高？
```

### 7. Arithmetic Intensity

```text
每搬一些 Bytes 能做多少 FLOPs？
Tiling 是否显著减少 HBM traffic？
AI 是否足够高？
```

### 8. Resource Usage / Occupancy

```text
Tile size 是否太大？
Shared Memory / Block 是否太高？
Registers / Thread 是否太多？
Threads / Block 是否接近上限？
一个 SM 能驻留多少 Blocks / Warps？
latency hiding 是否足够？
```

### 9. Bottleneck 判断

```text
更可能 memory-bound 还是 compute-bound？
瓶颈在 HBM？
瓶颈在 Shared Memory bank conflict？
瓶颈在 occupancy？
瓶颈在 compute throughput？
```

---

## 24. Day4 应该保留下来的心智模型

### Bank 模型

```text
Bank 不是行，也不是列。
Bank 是 Shared Memory 地址映射到的硬件通道。
```

### Padding 模型

```text
tile[32][33]
不是为了多存数据
而是为了改变 stride
让按列访问时 Bank 编号错开
```

### Tiling 模型

```text
Tiling = 把大矩阵切成小块
Shared Memory = 存当前 A/B Tile
Threads = 合作搬 Tile，并重复使用 Tile
Register = 保存每个 Thread 自己的 partial sum
```

### MatMul reuse 模型

```text
A 的一个元素
→ 被同一 C Tile 中多个列方向输出使用

B 的一个元素
→ 被同一 C Tile 中多个行方向输出使用
```

### Shared Memory 边界模型

```text
Shared Memory 只属于一个 Block
Block 间不能直接共享 Shared Memory
跨 Block reuse 主要依赖 cache，而不是 shared memory
```

### 同步模型

```text
第一个 __syncthreads():
大家都搬完了吗？

第二个 __syncthreads():
大家都用完了吗？
```

---

## 25. Day4 收束

Day4 把前几天的知识串成了一条完整链：

```text
Operator
↓
Kernel
↓
Thread / Block / Warp
↓
Global Memory Coalescing
↓
Shared Memory
↓
Bank Conflict
↓
Tiling
↓
Data Reuse
↓
Register partial sum
↓
Arithmetic Intensity
↓
Memory-bound / Compute-bound 判断
```

到这里，基础 Tiled MatMul 的核心已经通了：

```text
一个 Block 负责一个 C Tile
多个 Threads 合作搬 A/B Tile
Shared Memory 保存当前 round 的 A/B Tile
每个 Thread 用 As[ty][k] * Bs[k][tx] 累加自己的 C[row][col]
K 方向多轮迭代
两个 __syncthreads() 保证 Shared Memory 不被提前读/覆盖
最后每个 Thread 写回自己的 C element
```

这就是 Day4 最重要的结论：

> 高性能 Kernel 优化的核心不是“用了某个高级语法”，而是沿着数据流判断：数据从哪里来、被谁用、用了几次、访问模式是否适合硬件、资源是否撑得住。

