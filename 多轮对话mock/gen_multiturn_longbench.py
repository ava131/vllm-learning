#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成 LongBench v1 格式的多轮 agent 对话数据集（纯压测用）。

场景库放在 scenarios.json（可按题材自行增删），本脚本只负责：
  取场景 → 按 style 套用问句/回答模板 → 累积多轮上下文 → 补到目标长度 → 写 jsonl。

每条 jsonl = 一次请求，字段严格对齐 LongBench v1：
    input / context / answers / length / dataset / language / all_classes / _id

第 t 轮请求的 context 逐字节包含前 t-1 轮的 (user, assistant) 原文，
且所有请求共享同一段 system prompt，按文件顺序发送即可命中前缀缓存。

用法：
    python3 gen_multiturn_longbench.py --num-turns 3 --users-per-turn 20
    python3 gen_multiturn_longbench.py --num-turns 5 --users-per-turn 20,20,20,20,10 \
        --user-chars 150 --answer-chars 400 --out datasets/agent_mt.jsonl
"""

import argparse
import json
import os
import random
import sys
from typing import Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------------------------------- #
# 按 style 分类的模板（场景只提供名词，句式由这里决定）
# --------------------------------------------------------------------------- #

STYLES_ZH = {
    "diagnose": {
        "sec": ("证据：", "建议："),
        "head": "结论：关于{subject}，异常主要集中在{focus}，与 {entity} 在 {when} 前后的变化吻合。",
        "q": [
            "帮我看下{subject}，{when}开始不太对，先给个初步结论。",
            "继续，把{focus}也查一下，说明下判断依据。",
            "结合上面的结论，给我一份{deliverable}，要能直接发到{who}。",
            "再补充一点{focus}的情况，尤其是{entity}那边。",
        ],
        "fact": "- {when} {entity} {fname}={value}{unit}（基线 {base}{unit}，已超阈值）",
        "gen": "- 核对 {entity}：{focus} 相关指标与告警时间基本对齐。",
        "advice": "- 第 {i} 步：对 {entity} 执行限流并观察 10 分钟，回滚入口已就绪。",
    },
    "solve": {
        "sec": ("推导：", "易错点："),
        "head": "答案：关键在{focus}，先把 {entity} 的关系理清楚。",
        "q": [
            "帮我一步步推一下{subject}。",
            "第二问也卡住了，{focus}这块再讲细一点。",
            "把过程整理成{deliverable}，我能照着写。",
            "顺便说说{focus}常见的坑。",
        ],
        "fact": "- 第 {i} 步：由 {fname}={value}{unit} 联立方程，消去一个未知量。",
        "gen": "- 第 {i} 步：检查 {entity} 的取值范围，确认没有漏解。",
        "advice": "- 易错点 {i}：{focus}容易在单位或符号上出错，最后要回代验证。",
    },
    "explain": {
        "sec": ("例子：", "延伸："),
        "head": "要点：关于{subject}，核心是{focus}，可以从 {entity} 入手。",
        "q": [
            "解释一下{subject}，最好举个例子。",
            "{focus}这块还是不太懂，换个说法讲讲。",
            "整理成{deliverable}，能直接给{who}看。",
            "再补充点背景，说说它和{focus}的关系。",
        ],
        "fact": "- {entity} 的 {fname} 约为 {value}{unit}，可以用来说明{focus}。",
        "gen": "- 补充一个例子：{entity} 的实际情况比抽象定义更直观。",
        "advice": "- 延伸 {i}：想深入可以顺着 {entity} 查相关章节，先搭框架。",
    },
    "create": {
        "sec": ("提纲：", "润色："),
        "head": "思路：以{focus}为主线，用 {entity} 作为支撑。",
        "q": [
            "帮我写{subject}，先出个思路。",
            "结构再调一下，{focus}这块要加强。",
            "按这个思路给我一份{deliverable}。",
            "语言再润色一下，别太书面。",
        ],
        "fact": "- 素材：{entity} 的 {fname} 是 {value}{unit}，可以作为开头的事实引入。",
        "gen": "- 段落 {i}：围绕{focus}展开，用 {entity} 举例。",
        "advice": "- 润色 {i}：把 {entity} 那句改成短句，节奏会更稳。",
    },
    "plan": {
        "sec": ("步骤：", "风险："),
        "head": "目标：先保证{focus}，{entity} 的投入按周分配。",
        "q": [
            "帮我安排{subject}，给个可执行的计划。",
            "{focus}这块再细化，时间安排也说下。",
            "整理成{deliverable}，我要照着执行。",
            "中途有变动怎么调整？",
        ],
        "fact": "- 第 {i} 周：{entity} 完成{focus}的基础部分，阶段指标 {fname}={value}{unit}。",
        "gen": "- 第 {i} 周：先做 {entity}，把{focus}拆成可检查的小任务。",
        "advice": "- 风险 {i}：{focus}容易中断，留出 20% 的缓冲时间。",
    },
    "compare": {
        "sec": ("对比：", "结论："),
        "head": "先看维度：{focus}决定取舍，{entity} 是参照。",
        "q": [
            "对比一下{subject}，给个结论。",
            "{focus}这个维度再说细一点。",
            "整理成{deliverable}，能直接拿去讨论。",
            "如果条件变了，结论还成立吗？",
        ],
        "fact": "- 维度 {i}：{entity} 的 {fname} 为 {value}{unit}，优劣差距明显。",
        "gen": "- 维度 {i}：比较 {entity} 在{focus}上的表现，差距不在数量。",
        "advice": "- 适用 {i}：短期看{focus}，长期看 {entity} 的维护成本。",
    },
    "advise": {
        "sec": ("建议：", "注意："),
        "head": "现状：问题大概率出在{focus}，{entity} 是首要检查对象。",
        "q": [
            "关于{subject}，最近有点困扰，给点建议。",
            "试过一些办法，{focus}这块还是没改善。",
            "整理成{deliverable}，我照着做。",
            "有什么要特别注意的？",
        ],
        "fact": "- 记录：{fname} 目前约 {value}{unit}，先作为基线。",
        "gen": "- 观察：{entity} 的{focus}变化最明显，从这里入手。",
        "advice": "- 建议 {i}：先调整{focus}，观察一周再决定是否加码。",
    },
    "review": {
        "sec": ("问题：", "修改建议："),
        "head": "整体可用，主要问题集中在{focus}，{entity} 需要调整。",
        "q": [
            "帮我看看{subject}，有没有问题。",
            "{focus}这块再挑挑毛病。",
            "按你的意见改一版，给我{deliverable}。",
            "还有哪些容易忽略的地方？",
        ],
        "fact": "- 问题 {i}：{focus}缺少数据支撑，补一个 {fname}={value}{unit} 的例子。",
        "gen": "- 问题 {i}：{entity} 的{focus}表述含糊，改成可验证的说法。",
        "advice": "- 检查项 {i}：改完通读一遍，确认{focus}前后一致。",
    },
}


STYLES_EN = {
    "diagnose": {
        "sec": ("Evidence:", "Recommendation:"),
        "head": "Conclusion: the anomaly in {subject} is concentrated in {focus}, matching changes on {entity} around {when}.",
        "q": [
            "Could you look into {subject}? It started to look off at {when}; give me an initial read.",
            "Keep going and check {focus}, with the reasoning behind it.",
            "Based on that, draft a {deliverable} I can send to {who}.",
            "One more thing about {focus}, especially on {entity}.",
        ],
        "fact": "- {when} {entity} {fname}={value}{unit} (baseline {base}{unit}, above threshold)",
        "gen": "- Cross-check {entity}: {focus} lines up with the alert window.",
        "advice": "- Step {i}: rate-limit {entity} and watch for 10 minutes; rollback is ready.",
    },
    "solve": {
        "sec": ("Derivation:", "Pitfalls:"),
        "head": "Answer: the key is {focus}; sort out the relation around {entity} first.",
        "q": [
            "Walk me through {subject} step by step.",
            "The second part is stuck too; expand on {focus}.",
            "Turn it into {deliverable} I can follow.",
            "Also mention the usual traps in {focus}.",
        ],
        "fact": "- Step {i}: from {fname}={value}{unit}, set up the equation and remove one unknown.",
        "gen": "- Step {i}: check the range of {entity} to make sure no solution is dropped.",
        "advice": "- Pitfall {i}: {focus} often breaks on units or signs; substitute back at the end.",
    },
    "explain": {
        "sec": ("Example:", "Going further:"),
        "head": "Key idea: for {subject}, the core is {focus}, and {entity} is a good entry point.",
        "q": [
            "Explain {subject}, ideally with an example.",
            "I still do not get {focus}; say it another way.",
            "Turn it into {deliverable} I can share with {who}.",
            "Add some background: how does it relate to {focus}?",
        ],
        "fact": "- {entity} has {fname} around {value}{unit}, which illustrates {focus}.",
        "gen": "- Another example: the actual case of {entity} is easier to grasp than a definition.",
        "advice": "- Going further {i}: follow {entity} into the related chapter and build the frame first.",
    },
    "create": {
        "sec": ("Outline:", "Polish:"),
        "head": "Approach: make {focus} the through-line and use {entity} as support.",
        "q": [
            "Draft {subject}; start with an outline.",
            "Adjust the structure; strengthen {focus}.",
            "Turn that outline into {deliverable}.",
            "Polish the language; keep it less formal.",
        ],
        "fact": "- Material: {entity} has {fname} of {value}{unit}; use it as an opening fact.",
        "gen": "- Paragraph {i}: develop {focus} with {entity} as the example.",
        "advice": "- Polish {i}: shorten the {entity} sentence for better rhythm.",
    },
    "plan": {
        "sec": ("Steps:", "Risks:"),
        "head": "Goal: secure {focus} first; spread the {entity} effort across weeks.",
        "q": [
            "Plan {subject} for me, something executable.",
            "Break down {focus} further, including the timing.",
            "Turn it into {deliverable} I can follow.",
            "What if things change midway?",
        ],
        "fact": "- Week {i}: {entity} finishes the basics of {focus}; milestone {fname}={value}{unit}.",
        "gen": "- Week {i}: start with {entity} and split {focus} into checkable tasks.",
        "advice": "- Risk {i}: {focus} tends to stall; keep a 20% buffer.",
    },
    "compare": {
        "sec": ("Comparison:", "Verdict:"),
        "head": "Dimensions first: {focus} decides the trade-off, with {entity} as the reference.",
        "q": [
            "Compare {subject} and give me a verdict.",
            "Expand on {focus} as a dimension.",
            "Turn it into {deliverable} for discussion.",
            "If conditions change, does the verdict still hold?",
        ],
        "fact": "- Dimension {i}: {entity} has {fname} of {value}{unit}; the gap is clear.",
        "gen": "- Dimension {i}: compare how {entity} performs on {focus}; the gap is not about volume.",
        "advice": "- Applicability {i}: short term favors {focus}; long term favors {entity} maintenance cost.",
    },
    "advise": {
        "sec": ("Recommendation:", "Watch out:"),
        "head": "Status: the issue most likely comes from {focus}; {entity} is the first thing to check.",
        "q": [
            "I have had trouble with {subject} lately; any advice?",
            "Tried a few things; {focus} has not improved.",
            "Turn it into {deliverable} I can follow.",
            "Anything I should watch out for?",
        ],
        "fact": "- Record: {fname} is about {value}{unit}; use it as the baseline.",
        "gen": "- Observation: {focus} around {entity} changes most; start there.",
        "advice": "- Recommendation {i}: adjust {focus} first, then reassess in a week.",
    },
    "review": {
        "sec": ("Issues:", "Fixes:"),
        "head": "Usable overall; the main issues center on {focus}, and {entity} needs rework.",
        "q": [
            "Take a look at {subject} and point out the issues.",
            "Pick apart {focus} a bit more.",
            "Revise per your notes and give me {deliverable}.",
            "What else is easy to miss?",
        ],
        "fact": "- Issue {i}: {focus} lacks evidence; add an example with {fname}={value}{unit}.",
        "gen": "- Issue {i}: the {focus} wording around {entity} is vague; make it verifiable.",
        "advice": "- Check {i}: after revising, read it once and confirm {focus} is consistent.",
    },
}


def get_styles(language: str) -> dict:
    return STYLES_ZH if language == "zh" else STYLES_EN

TIME_POOL = ["09:12", "10:05", "11:38", "13:20", "14:47", "16:02", "17:25", "18:40",
             "20:15", "21:03", "22:31", "23:58"]
DAY_POOL = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]

TAILS_ZH = [
    "补充信息：相关材料我放在附件里了，可能不全。",
    "补充：上次的结论我还留着，需要的话我贴上来。",
    "补充：时间比较紧，先给关键结论就行。",
    "补充：相关截图我整理好了，随消息一起发。",
]
FILLERS_ZH = [
    "另外，背景资料我按时间顺序整理过，方便你对照。",
    "还有一点，之前的记录里提到过类似情况，可以参考。",
    "如果信息不够，我可以再补一份更详细的说明。",
]
FILLERS_EN = [
    "I also sorted the background material by time for easier cross-checking.",
    "One more thing: earlier records mention a similar case worth comparing.",
    "If anything is missing I can send a more detailed write-up.",
]
TAILS_EN = [
    "Context: the supporting material is attached, it may be incomplete.",
    "Context: I still have the previous notes and can paste them if needed.",
    "Context: time is tight, the key conclusion is enough for now.",
    "Context: I have screenshots ready and will send them along.",
]


def unit_len(text: str, language: str) -> int:
    """LongBench 官方口径：中文按字符数，英文按词数。"""
    return len(text) if language == "zh" else len(text.split())


def trim_to(text: str, target: int, language: str) -> str:
    """把文本裁到接近 target，尽量在句末截断。"""
    if target <= 0 or unit_len(text, language) <= target:
        return text
    if language == "zh":
        cut = text[:target]
        for mark in ("。", "\n", "；"):
            pos = cut.rfind(mark)
            if pos >= int(target * 0.6):
                return cut[:pos + len(mark)].rstrip() + "…"
        return cut.rstrip() + "…"
    # 英文按词截断，但保留换行结构
    out, used = [], 0
    for line in text.split("\n"):
        words = line.split()
        if used + len(words) <= target:
            out.append(line)
            used += len(words)
            continue
        remain = target - used
        if remain > 0:
            out.append(" ".join(words[:remain]))
        break
    return "\n".join(out).rstrip() + " ..."


def build_system_prompt(language: str, target: int) -> str:
    """全局共享、与场景无关的 system prompt。"""
    if language == "zh":
        head = ("你是一个知识面较广的智能助手，服务于「示例科技」。用户会围绕学习、工作与生活"
                "提出问题，你需要给出结论、依据与可执行的建议。\n\n"
                "工作守则：\n"
                "1. 先给结论，再给依据，最后给建议；\n"
                "2. 不确定的内容要说明不确定性，不臆造事实；\n"
                "3. 涉及操作或变更时，说明影响范围与回退方式；\n"
                "4. 回答保持简洁，避免重复用户原话。\n\n"
                "可用能力：")
    else:
        head = ("You are a general-purpose assistant at Example Corp. Users ask about study, "
                "work and daily life; answer with a conclusion, the evidence behind it and "
                "actionable advice.\n\n"
                "Rules:\n"
                "1. Conclusion first, then evidence, then recommendation.\n"
                "2. State uncertainty instead of inventing facts.\n"
                "3. For any action, mention blast radius and how to revert.\n\n"
                "Available capabilities:\n")

    items_zh = [
        "联网检索：可查资料并给出来源链接。",
        "计算器：支持四则运算、方程与单位换算。",
        "日历提醒：可创建日程、设置提醒。",
        "翻译：支持中英互译并保留术语表。",
        "代码执行：可运行 Python 片段验证结论。",
        "百科知识：覆盖历史、地理、天文、艺术等常识。",
        "天气查询：可查未来 7 天天气与空气质量。",
        "地图路线：可规划出行路线与耗时。",
        "文档写作：可起草、改写与润色。",
        "表格处理：可做汇总、透视与公式。",
        "图片理解：可读取截图中的文字与图表。",
        "待办管理：可拆解任务并跟踪进度。",
    ]
    items_en = [
        "web search with sources",
        "calculator for arithmetic, equations and units",
        "calendar and reminders",
        "translation with a glossary",
        "code execution to verify results",
        "general knowledge on history, geography, astronomy and art",
        "7-day weather and air quality",
        "route planning with travel time",
        "document drafting and editing",
        "spreadsheet summaries and formulas",
        "image understanding for screenshots and charts",
        "task breakdown and progress tracking",
    ]
    items = items_zh if language == "zh" else items_en

    lines = [head]
    i = 0
    while unit_len("\n".join(lines), language) < target and i < 60:
        lines.append("- " + items[i % len(items)])
        i += 1
    return trim_to("\n".join(lines), target, language)


def load_scenarios(path: str, language: str) -> List[dict]:
    if not os.path.exists(path):
        raise SystemExit("找不到场景文件: %s" % path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or language not in data:
        raise SystemExit("场景文件里没有 %r 语言的场景（现有: %s）"
                         % (language, list(data.keys()) if isinstance(data, dict) else type(data)))
    scenarios = data[language]
    if not isinstance(scenarios, list) or not scenarios:
        raise SystemExit("场景文件里 %r 的场景为空" % language)

    seen = set()
    for i, sc in enumerate(scenarios):
        for key in ("id", "domain", "style", "subject", "focus", "entities",
                    "deliverable", "who"):
            if key not in sc:
                raise SystemExit("第 %d 个场景缺少字段 %r: %r" % (i, key, sc.get("id")))
        if sc["style"] not in STYLES_ZH:
            raise SystemExit("场景 %s 的 style=%r 不在支持列表: %s"
                             % (sc["id"], sc["style"], list(STYLES_ZH)))
        if sc["id"] in seen:
            raise SystemExit("场景 id 重复: %s" % sc["id"])
        seen.add(sc["id"])
        if len(sc["focus"]) < 2 or len(sc["entities"]) < 2:
            raise SystemExit("场景 %s 的 focus/entities 至少各要 2 个" % sc["id"])
        facts = sc.get("facts") or []
        for fact in facts:
            if len(fact) != 4:
                raise SystemExit("场景 %s 的 facts 项必须是 [名称, 单位, 下限, 上限]: %r"
                                 % (sc["id"], fact))
    return scenarios


def render(tpl: str, sc: dict, rng: random.Random, when: str, step: int,
           rot: int, fact: Optional[List] = None) -> str:
    """step 用于序号（第几步），rot 用于轮换 focus/entity，避免整段重复。"""
    ctx = {
        "subject": sc["subject"],
        "focus": sc["focus"][rot % len(sc["focus"])],
        "entity": sc["entities"][rot % len(sc["entities"])],
        "deliverable": sc["deliverable"],
        "who": sc["who"],
        "when": when,
        "i": step,
    }
    if fact:
        name, unit, lo, hi = fact
        lo, hi = int(lo), int(hi)
        base = lo + (hi - lo) // 3
        value = rng.randint(base + 1, hi) if base < hi else hi
        ctx.update({"fname": name, "unit": unit, "value": value, "base": base})
    return tpl.format(**ctx)


def build_question(sc: dict, language: str, turn: int, target: int,
                   rng: random.Random, when: str) -> str:
    style = get_styles(language)[sc["style"]]
    head = render(style["q"][(turn - 1) % len(style["q"])], sc, rng, when, turn, turn - 1)
    tails = TAILS_ZH if language == "zh" else TAILS_EN
    tail = render(tails[(turn - 1) % len(tails)], sc, rng, when, turn, turn - 1)
    text = head + "\n" + tail
    fillers = FILLERS_ZH if language == "zh" else FILLERS_EN
    k = 0
    while unit_len(text, language) < target and k < 4:
        text += "\n" + fillers[k % len(fillers)]
        k += 1
    return trim_to(text, target, language)


def build_answer(sc: dict, language: str, turn: int, target: int,
                 rng: random.Random, when: str) -> str:
    style = get_styles(language)[sc["style"]]
    facts = sc.get("facts") or []
    head = render(style["head"], sc, rng, when, turn, turn - 1)
    sec1, sec2 = style["sec"]

    # 先长证据段到约 60% 预算，再长建议段，保证两个小节都能出现
    evidence: List[str] = []
    budget = max(int(target * 0.6), 60)
    i = 0
    while i < 30:
        if facts:
            evidence.append(render(style["fact"], sc, rng, when, i + 1, turn - 1 + i,
                                   facts[i % len(facts)]))
        evidence.append(render(style["gen"], sc, rng, when, i + 1, turn - 1 + i))
        body = "{head}\n\n{sec1}\n{ev}".format(head=head, sec1=sec1,
                                                 ev="\n".join(evidence))
        if unit_len(body, language) >= budget:
            break
        i += 1

    advice: List[str] = []
    j = 0
    while j < 6:
        advice.append(render(style["advice"], sc, rng, when, j + 1, turn - 1 + j))
        text = "{head}\n\n{sec1}\n{ev}\n\n{sec2}\n{ad}".format(
            head=head, sec1=sec1, ev="\n".join(evidence),
            sec2=sec2, ad="\n".join(advice))
        if unit_len(text, language) >= int(target * 0.9):
            break
        j += 1

    tail = ("备注：以上只针对当前场景，换条件后结论可能需要调整。"
            if language == "zh" else
            "Note: this holds for the current setup; changing conditions may change the conclusion.")

    def compose() -> str:
        return "{head}\n\n{sec1}\n{ev}\n\n{sec2}\n{ad}\n\n{tail}".format(
            head=head, sec1=sec1, ev="\n".join(evidence),
            sec2=sec2, ad="\n".join(advice), tail=tail)

    text = compose()
    # 超长时按"整行"回退，保证两个小节都还在，而不是把后半段直接截掉
    while unit_len(text, language) > target:
        if len(advice) > 1:
            advice.pop()
        elif len(evidence) > 2:
            evidence.pop()
        else:
            break
        text = compose()
    return trim_to(text, target, language)


def parse_users_per_turn(spec: str, num_turns: int) -> List[int]:
    raw = [s.strip() for s in spec.split(",") if s.strip()]
    if not raw:
        raise SystemExit("--users-per-turn 不能为空")
    counts = []
    for item in raw:
        try:
            n = int(item)
        except ValueError:
            raise SystemExit("--users-per-turn 只接受整数或逗号分隔的整数，收到: %r" % item)
        if n <= 0:
            raise SystemExit("--users-per-turn 的每个值必须 > 0，收到: %d" % n)
        counts.append(n)
    if len(counts) == 1:
        counts = counts * num_turns
    elif len(counts) < num_turns:
        counts = counts + [counts[-1]] * (num_turns - len(counts))
    else:
        counts = counts[:num_turns]
    return counts


def generate(args: argparse.Namespace, scenarios: List[dict],
             counts: List[int]) -> List[dict]:
    rng = random.Random(args.seed)
    pick = random.Random(args.seed ^ 0x5EED)
    shuffled = list(scenarios)
    pick.shuffle(shuffled)

    system = build_system_prompt(args.language, args.system_chars)
    histories: Dict[int, List[str]] = {}
    records: List[dict] = []

    for turn in range(1, args.num_turns + 1):
        for user in range(counts[turn - 1]):
            sc = shuffled[user % len(shuffled)]
            when = "{d} {t}".format(d=rng.choice(DAY_POOL), t=rng.choice(TIME_POOL))
            question = build_question(sc, args.language, turn,
                                      args.user_chars, rng, when)
            answer = build_answer(sc, args.language, turn,
                                  args.answer_chars, rng, when)

            parts = histories.get(user, [])
            context = args.sep.join([system] + parts) if parts else system
            records.append({
                "input": question,
                "context": context,
                "answers": [answer],
                "length": (unit_len(question, args.language)
                           + unit_len(context, args.language)
                           + unit_len(answer, args.language)),
                "dataset": args.dataset,
                "language": args.language,
                "all_classes": None,
                "_id": "u{uid:02d}_t{turn}".format(uid=user, turn=turn),
            })
            histories[user] = parts + [question, answer]
    return records


def verify(records: List[dict], args: argparse.Namespace, counts: List[int]) -> None:
    expected_keys = {"input", "context", "answers", "length", "dataset",
                     "language", "all_classes", "_id"}
    assert len(records) == sum(counts), \
        "记录数 %d != 期望 %d" % (len(records), sum(counts))

    seen = set()
    by_user: Dict[int, List[dict]] = {}
    for r in records:
        assert set(r.keys()) == expected_keys, \
            "字段不匹配: %s" % (set(r.keys()) ^ expected_keys)
        assert isinstance(r["answers"], list) and r["answers"], "answers 必须是非空 list"
        assert isinstance(r["length"], int) and r["length"] > 0, "length 必须是正整数"
        assert r["all_classes"] is None, "all_classes 应为 null"
        assert r["_id"] not in seen, "重复 _id: %s" % r["_id"]
        seen.add(r["_id"])
        uid = int(r["_id"].split("_")[0][1:])
        by_user.setdefault(uid, []).append(r)

    for uid, items in by_user.items():
        for prev, cur in zip(items, items[1:]):
            expected = prev["context"] + args.sep + prev["input"] + args.sep + prev["answers"][0]
            assert cur["context"] == expected, "user %d 前缀链断裂: %s" % (uid, cur["_id"])


def summarize(records: List[dict], args: argparse.Namespace, counts: List[int],
              scenarios: List[dict]) -> None:
    print("\n%-6s %-8s %-14s %-11s %-12s %-12s" % (
        "turn", "reqs", "ctx_len(avg)", "input(avg)", "answer(avg)", "length(avg)"),
        file=sys.stderr)
    idx = 0
    for turn, n in enumerate(counts, start=1):
        chunk = records[idx:idx + n]
        idx += n
        ctx = sum(unit_len(r["context"], args.language) for r in chunk) / n
        inp = sum(unit_len(r["input"], args.language) for r in chunk) / n
        ans = sum(unit_len(r["answers"][0], args.language) for r in chunk) / n
        ln = sum(r["length"] for r in chunk) / n
        print("t%-5d %-8d %-14.0f %-11.0f %-12.0f %-12.0f"
              % (turn, n, ctx, inp, ans, ln), file=sys.stderr)

    unit = {"zh": "字符", "en": "词"}[args.language]
    used = len({r["_id"].split("_")[0] for r in records})
    print("\n总计 %d 条 / %d 个用户，场景库 %d 个（本次用到 %d 个），"
          "长度单位%s，dataset=%s，language=%s，分隔符=%r"
          % (len(records), used, len(scenarios), min(used, len(scenarios)),
             unit, args.dataset, args.language, args.sep), file=sys.stderr)
    if len(scenarios) < max(counts):
        print("⚠️  场景数(%d) < 单轮最大用户数(%d)，会出现复用"
              % (len(scenarios), max(counts)), file=sys.stderr)


def main() -> None:
    p = argparse.ArgumentParser(
        description="生成 LongBench v1 格式的多轮 agent 对话 jsonl（压测用）")
    p.add_argument("--num-turns", type=int, default=3, help="对话轮数（默认 3）")
    p.add_argument("--users-per-turn", default="20",
                   help="每轮的用户/请求数，整数或逗号分隔（如 20 或 20,20,10），默认 20")
    p.add_argument("--scenarios", default=os.path.join(HERE, "scenarios.json"),
                   help="场景库 json 路径（默认脚本同目录的 scenarios.json）")
    p.add_argument("--system-chars", type=int, default=600,
                   help="system prompt 目标长度（默认 600）")
    p.add_argument("--user-chars", type=int, default=120,
                   help="每轮用户提问目标长度（默认 120）")
    p.add_argument("--answer-chars", type=int, default=300,
                   help="每轮助手回答目标长度（默认 300）")
    p.add_argument("--language", choices=["zh", "en"], default="zh")
    p.add_argument("--dataset", default="agent_mt", help="写入 dataset 字段的名字")
    p.add_argument("--sep", default="\n\n", help="消息之间的分隔符（默认两个换行）")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="agent_mt.jsonl", help="输出 jsonl 路径")
    p.add_argument("--manifest", default=None, help="可选：额外输出映射清单 json")
    p.add_argument("--no-verify", action="store_true", help="跳过写后校验")
    args = p.parse_args()

    if args.num_turns <= 0:
        raise SystemExit("--num-turns 必须 > 0")

    counts = parse_users_per_turn(args.users_per_turn, args.num_turns)
    scenarios = load_scenarios(args.scenarios, args.language)
    if args.system_chars <= 0 or args.user_chars <= 0 or args.answer_chars <= 0:
        raise SystemExit("--system-chars / --user-chars / --answer-chars 必须 > 0")

    records = generate(args, scenarios, counts)
    if not args.no_verify:
        verify(records, args, counts)

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    if args.manifest:
        with open(args.manifest, "w", encoding="utf-8") as f:
            json.dump({
                "dataset": args.dataset,
                "language": args.language,
                "num_turns": args.num_turns,
                "users_per_turn": counts,
                "scenario_file": os.path.abspath(args.scenarios),
                "seed": args.seed,
                "records": [{"line": i + 1, "id": r["_id"], "length": r["length"]}
                            for i, r in enumerate(records)],
            }, f, ensure_ascii=False, indent=2)

    summarize(records, args, counts, scenarios)
    print("已写入 %s（%d 行）" % (args.out, len(records)), file=sys.stderr)


if __name__ == "__main__":
    main()
