# 多轮对话 mock 数据集

生成 **LongBench v1 格式**的多轮对话 jsonl，用于推理服务的**纯性能压测**（前缀缓存命中、批量调度、KV 占用），不用于精度评测。

- 每条 jsonl = 一次请求，字段与 LongBench v1 官方完全对齐（8 个字段）
- 所有请求共享同一段 system prompt（逐字节一致）
- 第 t 轮请求的 `context` 逐字节包含前 t-1 轮的问答原文，按文件顺序发送即可命中前缀缓存
- 内容由「场景库 + 模板」合成，零依赖、可复现，不联网、不调用模型

## 目录结构

```
多轮对话mock/
├── README.md                     # 本文件
├── gen_multiturn_longbench.py    # 生成脚本（Python 3.8+，仅标准库）
├── scenarios.json                # 场景库：zh 56 个 + en 8 个
└── datasets/
    └── agent_mt_3x20.jsonl       # 示例产物：3 轮 × 每轮 20 用户 = 60 条
```

## 快速开始

```bash
# 默认：3 轮 × 每轮 20 用户 = 60 条，输出 ./agent_mt.jsonl
python3 gen_multiturn_longbench.py

# 推荐：直接生成到 datasets/
python3 gen_multiturn_longbench.py --num-turns 3 --users-per-turn 20 \
    --out datasets/agent_mt_3x20.jsonl

# 每轮人数不同（第 3 轮只剩前 10 个用户）
python3 gen_multiturn_longbench.py --num-turns 3 --users-per-turn 20,20,10 \
    --out datasets/agent_mt_20_20_10.jsonl

# 拉长上下文（每条约 2k tokens 以上），加大回答长度
python3 gen_multiturn_longbench.py --num-turns 5 --users-per-turn 20 \
    --system-chars 2000 --user-chars 300 --answer-chars 800 \
    --out datasets/agent_mt_big.jsonl

# 英文
python3 gen_multiturn_longbench.py --language en --num-turns 2 \
    --users-per-turn 10 --out datasets/agent_mt_en.jsonl
```

脚本每次写完都会**自校验**：字段集合、行数、`_id` 唯一性、前缀链是否逐字节衔接、`answers`/`length`/`all_classes` 类型。任一不满足直接报错退出（退出码非 0）。

## 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--num-turns` | `3` | 对话轮数 |
| `--users-per-turn` | `20` | 每轮的用户/请求数。单个整数会套用到每一轮；也可写成 `20,20,10`（长度不足时用最后一个值补齐，超长时截断） |
| `--scenarios` | 脚本同目录 `scenarios.json` | 场景库路径 |
| `--system-chars` | `600` | system prompt 目标长度 |
| `--user-chars` | `120` | 每轮用户提问目标长度 |
| `--answer-chars` | `300` | 每轮助手回答目标长度 |
| `--language` | `zh` | `zh` 按字符计长度，`en` 按词计长度 |
| `--dataset` | `agent_mt` | 写入 `dataset` 字段的值 |
| `--sep` | `\n\n` | 消息分隔符（影响前缀拼接方式） |
| `--seed` | `42` | 随机种子；同 seed 输出逐字节一致 |
| `--out` | `agent_mt.jsonl` | 输出路径（目录不存在会自动创建） |
| `--manifest` | 无 | 额外输出一份行号/`_id` 映射 json |
| `--no-verify` | 关闭 | 跳过写后校验 |

长度类参数的语义是「目标值」：素材不够会循环铺长，超了会在句末（`。`/换行）处截断，因此实际长度略小于等于目标值。

## 输出格式

每行一个 JSON 对象，字段与 LongBench v1 一致（顺序也保持一致）：

```json
{
  "input": "帮我看下payment-api 的 5xx 比例，周六 10:05开始不太对，先给个初步结论。\n补充信息：...",
  "context": "你是一个知识面较广的智能助手，服务于「示例科技」。...\n\n<第1轮问题>\n\n<第1轮回答>",
  "answers": ["结论：关于payment-api 的 5xx 比例，异常主要集中在...\n\n证据：\n- ...\n\n建议：\n- ..."],
  "length": 988,
  "dataset": "agent_mt",
  "language": "zh",
  "all_classes": null,
  "_id": "u00_t1"
}
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `input` | string | 本轮用户提问（对应 LongBench 的短输入） |
| `context` | string | system prompt + 前 t-1 轮对话；第 1 轮就是纯 system |
| `answers` | list[string] | 占位回答，**不是标准答案**（压测不看精度） |
| `length` | int | `input + context + answers` 的长度，中文按字符、英文按词 |
| `dataset` | string | 数据集名，默认 `agent_mt` |
| `language` | string | `zh` / `en` |
| `all_classes` | null | 非分类任务固定为 `null` |
| `_id` | string | `u<用户号>_t<轮次>`，如 `u07_t2` |

## 顺序与语义（重点）

**文件按轮次优先排序**：先第 1 轮的 20 条，再第 2 轮的 20 条，再第 3 轮的 20 条。直接按行顺序发送，就对应「bs 0–20 / 20–40 / 40–60」。

**system prompt 全局共享**：所有请求的前 N 个字符（默认约 590 字符）完全一致；从该用户第一个问题开始分叉。

**前缀链保证**：对同一用户，

```
context(t) == context(t-1) + sep + input(t-1) + sep + answers(t-1)[0]
```

所以第 2、3 轮能命中第 1 轮建立的 KV 前缀。注意：实际推理模板必须把 `context` 放在 `input` 之前（LongBench 官方模板都是这个顺序），并且不要在中途重新排版，否则前缀就断了。

### `--users-per-turn` 的不同情况

用户编号从 `u00` 开始，第 t 轮参与的是「编号 < 该轮人数」的用户。

| 写法 | 总条数 | 结果 |
|---|---|---|
| `20,20,20` | 60 | 20 个用户各 3 轮，`u00_t1 … u19_t3` |
| `20,20,10` | 50 | `u00`–`u09` 有 3 轮；`u10`–`u19` 停在第 2 轮。最后 10 行是 `u00_t3 … u09_t3` |
| `5,8,3` | 16 | 共 8 个用户：`u00`–`u02` 3 轮，`u03`/`u04` 2 轮，`u05`–`u07` 从第 2 轮才开始（context 只有 system，是新会话） |
| `20,10,20` | 50 | `u10`–`u19` 第 2 轮缺席、第 3 轮回归时**会接上自己第 1 轮的历史**（中间空一轮），不会重开新会话 |

也就是说：**人数减少 = 末尾用户提前结束；人数增多 = 新增编号从该轮新开会话；先减后增 = 回归用户续接旧历史**。如果你要「一旦缺席永不回归」，需要在 `generate()` 里加策略，当前默认是上面的语义。

缺席本身对压测有意义：这些用户的 KV 块会被逐步挤出，回归时只能命中 system 段，正好可以观察缓存淘汰行为。

## 场景库 `scenarios.json`

结构是 `{"zh": [...], "en": [...]}`，中文 56 个场景覆盖 33 个领域（数理化生、语文、历史、法律政策、地理、天文、艺术、生活、健康、理财、职场、技术、办公、教学、围棋/篮球/汽车/园艺……），英文 8 个。

单个场景：

```json
{
  "id": "cn_poetry",
  "domain": "语文",
  "style": "explain",
  "subject": "这首唐诗的鉴赏",
  "focus": ["意象选择", "炼字", "情感表达"],
  "entities": ["王维", "边塞", "月亮"],
  "facts": [["字数", "字", 20, 60]],
  "deliverable": "鉴赏思路",
  "who": "语文老师"
}
```

| 字段 | 必填 | 说明 |
|---|---|---|
| `id` | 是 | 唯一标识 |
| `domain` | 是 | 领域，仅用于描述 |
| `style` | 是 | 对话风格，决定问句句式与回答结构：`diagnose` / `solve` / `explain` / `create` / `plan` / `compare` / `advise` / `review` |
| `subject` | 是 | 话题（填进问句主语） |
| `focus` | 是 | ≥2 个关注点，按轮次轮换，制造多轮递进 |
| `entities` | 是 | ≥2 个实体，按轮次轮换 |
| `facts` | 否 | `[名称, 单位, 下限, 上限]`，有就生成带数值的证据行 |
| `deliverable` | 是 | 第 3 轮要产出的东西（如「故障复盘」） |
| `who` | 是 | 产出给谁看 |

加场景只要往 `zh` 数组里加一条，脚本会自动校验重复 `id`、缺字段、非法 `style`、`facts` 格式。用户与场景的对应关系是「按 `--seed` 打乱后取第 `i` 个」，因此 20 个用户一定拿到 20 个不同场景；单轮人数超过场景数时会复用并打印警告。

## 接到 AISBench

AISBench 的 LongBench 任务走 `load_dataset(path=<LongBench 目录>, name=<任务名>)`，再由目录里的 `LongBench.py` 读取 `data/<任务名>.jsonl`。所以：

1. 把生成的文件放到 `{aisbench}/ais_bench/datasets/LongBench/data/agent_mt.jsonl`；
2. 在 `LongBench.py` 的 `task_list` 里加一行 `"agent_mt"`（否则会报 `BuilderConfig 'agent_mt' not found`）；
3. 复制一份 `configs/datasets/longbench/longbench_2wikimqa/longbench_2wikimqa_gen.py`，把 `abbr` 改为 `agent_mt_gen`、`name` 改为 `'agent_mt'`，prompt 模板保留 `{context}` + `{input}` 即可；
4. 跑 perf：`ais_bench --models <vllm_api 任务> --datasets agent_mt_gen -m perf`。

注意事项：

- 这些配置是**单轮语义**（一行一个请求），多轮是靠 `context` 里带历史实现的，正好匹配「按顺序到达」；
- perf 模式不要开 shuffle / 随机采样，否则前缀命中会乱；
- `dataset` 字段用自定义名字时，**官方 LongBench 的 `eval.py` 会因为 `dataset2metric` 里没有这个 key 而报错**；纯压测不跑精度就没影响。

## 已知限制

- 文本是模板 + 场景库合成的，**通顺但事实是编的**（指标、时间、人名都是随机槽位），不能当知识库或评测集；
- `answers` 只是占位回答，不是标准答案；
- 同 `--seed` 输出完全一致，换 seed 会改变场景分配、数值和措辞；
- 生成器按字符/词计长度，不依赖 tokenizer；要精确 token 数请在 AISBench 或推理框架侧统计；
- 深色/浅色编码、文件编码统一为 UTF-8，`ensure_ascii=False`，可直接 `json.loads` 逐行读取。

## 手工验证

```bash
python3 - <<'EOF'
import json
recs = [json.loads(l) for l in open('datasets/agent_mt_3x20.jsonl', encoding='utf-8')]
keys = {"input","context","answers","length","dataset","language","all_classes","_id"}
assert all(set(r) == keys for r in recs)
by = {}
for r in recs: by.setdefault(r["_id"].split("_")[0], []).append(r)
for items in by.values():
    for p, c in zip(items, items[1:]):
        assert c["context"] == p["context"] + "\n\n" + p["input"] + "\n\n" + p["answers"][0]
print("ok:", len(recs), "lines")
EOF
```
