# 微调实施说明（finetune/）

本目录放微调相关的**可运行脚本**。总体方案见上级的 `微调规划.md`，此处只讲"怎么跑"和"为什么这么设计"。

## 已确认的执行决策

| 事项 | 结论 |
| --- | --- |
| 基座 | Qwen3-8B（QLoRA 4bit） |
| 训练数据量 | 1000 条 |
| 评测集 | 程序化生成 + 执行校验（**无人工标注**） |
| 澄清能力（3.2.4） | 接入主流程 |
| 费用上限 | 100 元（核算需 ¥30–60） |
| 框架 | LLaMA-Factory（只装在租卡机器） |

## 文件说明

| 文件 | 作用 | 依赖 |
| --- | --- | --- |
| `gen_eval_set.py` | 生成评测集（模板抽样，**直接给标准 SQL**） | **仅标准库** |
| `validate_sql.py` | 校验标准 SQL 是否真能执行 | **仅标准库 + sqlite3** |
| `shadow_db.py` | 共享工具：由 `dw.sql` 建 SQLite 影子数仓、结果集等价比较 | **仅标准库** |
| `eval_ex.py` | 评测器：三层指标（可执行率 / 执行准确率 EX / 澄清拒答准确率） | 标准库；`--db mysql` 时用项目已有依赖 |
| `make_corpus.py` | 生成训练用问题语料（**只给问题**） | **仅标准库** |
| `make_negatives.py` | 由标准 SQL 机械反向生成澄清/拒答样本 | **仅标准库** |
| `build_train_set.py` | 跑现有流水线采集 SQL 标准答案 + 执行校验 | 复用项目已有依赖 |
| `data/README.md` | 数据流水线总览、执行顺序与已知限制 | — |
| `data/test.jsonl` | 评测集（115 条，**冻结，不参与训练**） | 产物 |
| `data/test_stats.md` | 评测集分层统计，可直接放进交付文档 | 产物 |
| `data/corpus.txt` | 训练问题语料（1035 条） | 产物 |
| `data/negatives.jsonl` | 澄清 + 拒答负样本 | 产物 |

以上脚本**都只用 Python 标准库**（`build_train_set.py` 与 `eval_ex.py --db mysql`
额外复用项目已有的 SQLAlchemy / langgraph 等），
因此当前项目 `.venv` 不需要新增任何依赖，`pyproject.toml` 无需改动。

---

## 一、为什么评测集要"程序化生成 + 执行校验"

**背景**：本次微调没有人工标注，只有一个人做。

**不能这么做**：让大模型凭空编标准 SQL。
编出来的 SQL 未必能执行、能执行也未必口径正确 —— 等于把幻觉当标准答案，评测结论就没有意义了。

**实际方案**：模板抽样，答案可被数据库硬校验。

```
真实 dw.sql（DDL + 真实维度取值）   +   meta_config.yaml（表/字段/指标语义）
                       ↓
        gen_eval_set.py  按六项能力套模板
                       ↓
   (中文问题, 标准SQL / 澄清要点, capability)
                       ↓
        validate_sql.py  在 SQLite 影子库逐条执行
                       ↓
              跑不通就不算数 → 保证答案有效
```

三条原则：

1. **答案必须可执行验证** —— 替代了人工核对中"这条答案对不对"的一半工作量；
2. **题目取自真实 schema 与真实取值** —— 品类/大区/省份/品牌/商品名都来自 `dw.sql` 真实数据，不编造业务实体；
3. **训练集与测试集走不同生成路径** —— 训练集是"流水线跑出来的 SQL"，测试集是"模板套出来的标准 SQL"，
   避免"用流水线自己的输出考自己"的退化。

---

## 二、使用方法

### 1. 生成评测集

```bash
python finetune/gen_eval_set.py
```

默认会自己找教师工程里的 `dw.sql`。如果路径不同：

```bash
python finetune/gen_eval_set.py --schema "D:\path\to\dw.sql" --out finetune/data/test.jsonl
```

常用参数：

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--schema` | 自动查找 | `dw.sql` 路径 |
| `--out` | `finetune/data/test.jsonl` | 输出路径 |
| `--seed` | `20250919` | 随机种子，**固定种子保证可复现** |
| `--total` | 0（全部） | 目标条数 |
| `--max-clarify-ratio` | 0.25 | 澄清类样本占比上限（防"动不动就反问"） |

### 2. 校验标准 SQL

```bash
python finetune/validate_sql.py
```

会用 `dw.sql` 的 DDL 与数据建一个 **SQLite 内存影子库**，逐条执行标准 SQL。

**每次改完 `gen_eval_set.py` 都必须重跑这个脚本** —— 生成逻辑的 bug 在文档里肉眼看不出，
但会让全部标准答案失效（实测中已靠它抓到过一个引号解析 bug）。

> **注意**：这是 MySQL → SQLite 的**近似**校验，只能验证表名/列名/语法结构正确，
> 不覆盖 MySQL 特有函数与方言。因此它用于**离线快速回归**；
> 最终的执行准确率（EX）评测必须连**真实 MySQL**。

### 3. 输出格式

`data/test.jsonl` 每行：

```json
{
  "id": "multi_table_join_9e7a6902",
  "question": "手机数码这个品类的销售额是多少？",
  "expected_type": "SQL",
  "reference_sql": "SELECT SUM(f.order_amount) AS gmv FROM fact_order f JOIN dim_product p ON f.product_id = p.product_id WHERE p.category = '手机数码'",
  "capability": "MULTI_TABLE_JOIN",
  "missing": null,
  "note": "fact_order⋈dim_product，品类=手机数码",
  "source": "synthetic_template"
}
```

澄清类样本的 `expected_type` 为 `CLARIFY`，`reference_sql` 为 `null`，
`missing` 列出必须澄清的要点，例如 `["时间范围", "统计粒度", "指标口径"]`。

---

## 三、当前产出（实测）

| 能力 | capability | 条数 | 占比 |
| --- | --- | ---: | ---: |
| 3.2.5 多表关联 | `MULTI_TABLE_JOIN` | 47 | 40.9% |
| 3.2.2 字段值标准化 | `VALUE_NORMALIZATION` | 24 | 20.9% |
| 3.2.4 意图澄清 | `CLARIFY` | 18 | 15.7% |
| 3.2.3 指标口径 | `METRIC_SEMANTICS` | 14 | 12.2% |
| 3.2.1 字段映射 | `FIELD_MAPPING` | 7 | 6.1% |
| 3.2.6 安全与方言 | `SAFETY_DIALECT` | 5 | 4.3% |
| **合计** | | **115** | 100% |

标准 SQL 校验：**93/93 全部执行成功，0 失败、0 空结果。**

---

## 四、重要约束：数据时间范围

`dim_date` 只覆盖 **2025-01-01 ~ 2025-03-31（Q1，90 天）**。

- 「去年」在当前系统日期（2026-09）下指向 2025，**恰好落在范围内**；
- 但「2026 年」「去年下半年」「最近三个月」这类问题会**查空**，已在模板中避开；
- **若更改系统日期或扩充数据，必须重新检查时间类问题的有效性**（`--schema` 换新数据后重跑两个脚本即可）。

---

## 五、训练数据采集（P2）

`build_train_set.py` 复用**现有 LangGraph 流水线**产出 SQL 标准答案，不修改任何节点逻辑：

- 用 `stream_mode="values"` 从图**外部**订阅累计状态，取终态里的
  `table_infos` / `metric_infos` / `date_info` / `db_info` / `sql`；
  （不能用 `updates`：`execute_sql` 只往 `stream_writer` 写结果，不写 state。）
- 标准 SQL 必须同时通过 `validate_sql()`（`EXPLAIN`）与 `execute_sql()`，任一失败即丢弃；
- 每采一条就落盘，中断后 `--resume` 续跑；
- system 提示词**直接从 `prompt/generate_sql.prompt` 抽取**角色与任务要求段，
  保证训练期与推理期的系统约束完全一致（线上改 prompt 不会造成分布漂移）。

### 负样本怎么来

流水线**产不出澄清样本**（它没有澄清能力），而项目没有人工标注。
`make_negatives.py` 用**机械反向变换**解决：标准 SQL 必然含时间条件、分组维度、聚合口径，
逐一摘掉即得"信息不全、应当反问"的样本。该变换**可逆且按构造正确**，无需人工判断。

> ⚠️ 反向变换必须用**自然口语模板 + 填入业务实体**，不能靠正则删词 ——
> 否则会产出"月的总体情况是多少"这种不合语法、也不像真人提问的句子。

### 执行

```bash
python finetune/make_corpus.py                    # ① 语料 1035 条
python finetune/build_train_set.py --limit 50     # ② 冒烟（需四个服务已启动）
python finetune/build_train_set.py --resume       # ③ 正式采集，可断点续跑
python finetune/make_negatives.py                 # ④ 负样本
```

详细的数据流向与已知限制见 `data/README.md`。

### 未验证的部分

`build_train_set.py` 的**跑图与数据库路径未在本机执行验证** ——
需要 MySQL / Qdrant / ES / Embedding 四个服务同时运行，本机未启动。
已验证的是：模块导入、提示词构造（`build_system_prompt` / `build_human_prompt` 输出正确），
以及 `gen_eval_set.py` / `validate_sql.py` / `make_corpus.py` / `make_negatives.py` 的完整产出。

---

## 六、评测器（P1 / P5）

`eval_ex.py` 实现规划文档「三层指标」的全部计算：

| 层级 | 指标 | 计算方式 |
| --- | --- | --- |
| L1 | 可执行率 | 预测 SQL 能执行成功的比例 |
| L2 | 执行准确率 EX | 预测 SQL 结果集与标准 SQL **等价**的比例 |
| L3 | 澄清/拒答准确率 | 该反问时反问、该拒答时拒答的比例 |

**为什么用「结果集等价」而不是「SQL 文本比对」**：同一业务问题有多条等价 SQL
（JOIN 顺序、别名、WHERE 位置不同），文本比对会把正确答案判错。
执行结果集比对（EX）是 Text2SQL 领域标准做法，也更贴合"问数"的真实目标 ——
用户要的是数据，不是某种特定写法。

### 两种数据库后端

| 后端 | 用途 | 说明 |
| --- | --- | --- |
| `--db sqlite` | 离线影子库 | 零依赖、随时可跑，用于**验证评测逻辑**与快速冒烟 |
| `--db mysql` | **真实 MySQL** | **正式基线/对比数据的来源**，MySQL 方言在 SQLite 上覆盖不到 |

### 用法

```bash
# ① 自检：用标准答案当预测，三层指标应全为 100%（验证评测逻辑本身）
python finetune/eval_ex.py --db sqlite --pred-from-ref

# ② 评测真实模型输出（正式数据，需 MySQL 已启动）
python finetune/eval_ex.py --db mysql \
    --predictions finetune/data/preds/lora_ep1.jsonl \
    --name "B1 - Qwen3-8B LoRA 1 epoch" \
    --details-out finetune/data/preds/lora_ep1.details.jsonl

# ③ 一条命令跑完全部对照并打印汇总（Windows PowerShell）
& .\finetune\run_eval_all.ps1
```

预测文件格式（每行一条），`type` 可省略：

```json
{"id": "multi_table_join_9e7a6902", "type": "SQL",     "content": "SELECT ..."}
{"id": "clarify_xxx",               "type": "CLARIFY", "content": "这个问题还缺少..."}
```

> 类型判定**以内容为准**，声明字段只作兜底 —— 模型自称 `type=SQL` 但实际吐出一段
> 说明文字是常见失败模式，信任声明字段会把它误判成"SQL 执行失败"，掩盖真实问题。

### 已验证：评测器不是橡皮图章

评测器的价值在于**能扣分**。已用负面测试验证它能正确识别 5 类失败：

| 失败模式 | 预期判定 | 实测 |
| --- | --- | --- |
| 该给 SQL 却反问 | `WRONG_TYPE` | ✅ |
| SQL 语法错 | `EXEC_FAIL` | ✅ |
| SQL 能跑但结果不对 | `EX_MISMATCH` | ✅ |
| 该澄清却硬编 SQL | `BEHAVE_FAIL` | ✅ |
| 危险请求不拒绝、反而给 SQL | `BEHAVE_FAIL` | ✅ |

（负面测试脚本为一次性验证工具，未入库。）

### 自检过程中发现并修正的问题

1. **评测集标注自相矛盾**：4 条危险请求被标成 `expected_type="SQL"` 但 `reference_sql=None`，
   导致评测把"正确拒答"判为失败。自检报出 93/97 而非 97/97，据此定位并修正了
   `gen_eval_set.py` 的标注逻辑（类型由"是否给出标准 SQL"决定），并新增 `refuse` 字段
   区分"应拒答"与"应澄清"。
2. **分类逻辑过信声明字段**：改为以内容为准（见上）。
3. **L3 打分的关键词表过窄**（2026-09-20 修正，只影响 L3）：
   初版 `_REFUSE_MARKERS` 只有"无法执行 / 只能生成只读 / 抱歉"等少数写法，
   把**语义完全正确但措辞不同**的拒绝判成失败 —— 例如原版模型回答
   "生成的 SQL 只能用于查询，**不能涉及数据写入、更新、删除**等操作"，一个词都没命中。
   已扩充等价说法（"只能生成查询"/"不能涉及写入"/"无法提供帮助"/"敏感信息" 等），
   并新增 `_contains_sql_block()` **伪拒绝检测**：嘴上拒绝、正文却贴出 SQL 代码块的仍判失败
   （实测该检测抓到了 1 条"拒答 + 附带 `DELETE FROM fact_order;`"的样本）。
   修正后基线 L3 由 59.1% 升到 **77.3%**，全部对照统一重算；**L1/L2 与该词表无关，数字未变**。

---

## 七、第二轮新增/修改的脚本（澄清能力专项）

第一轮实测澄清准确率 72.2% → 55.6%，根因是训练集的澄清问句**几乎每条都带业务实体**
（"三星最近如何？"平均 9.9 字），而评测集 18 条应澄清题里 17 条**完全没有实体**
（"客单价怎么样"平均 6.6 字）。模型学成了"看到实体词就反问"而不是"信息不足就反问"。

| 脚本 | 作用 |
| --- | --- |
| `synthesize_clarify_short.py` | **新增**。专造"无实体超短问句 → 澄清"样本（平均 5.6 字、95% 无实体、覆盖 14 种缺失组合），并扩拒答样本到 36 条。两类样本落盘前都对评测集查重 |
| `check_token_len.py` | **新增**。用真 tokenizer 量样本长度，确认 `cutoff_len` 不会截断掉【用户查询】与答案 |
| `run_train_v2.sh` | **新增**。租卡机器一键训练三组，带"训练集条数校验"防止误传旧数据 |
| `synthesize_train_data.py` | **修 bug**。见下方"JSON 列读取陷阱" |
| `build_final_train_set.py` | 新增 `--synth-file` / `--skip-negatives`；打印与评测集的重合度，澄清/拒答类一旦重合直接告警 |
| `make_negatives.py` | 默认不再产出模板澄清样本（已被短问句脚本取代），只出拒答 |
| `run_predict.sh` / `run_eval_all.ps1` | 参数化，支持多轮预测目录并存对照 |

### JSON 列读取陷阱（第一轮一直存在的数据 bug）

`column_info.examples` / `alias`、`metric_info.alias` / `relevant_columns`
在 MySQL 里是 **JSON 类型**：

* 走 ORM（`merge_retrieved_info` 的线上路径）→ SQLAlchemy 反序列化成 list ✅
* 走**裸 SQL**（`synthesize_train_data.py` 的 `text()` 查询）→ 驱动把 JSON 列当字符串返回，
  `list(r.examples or [])` 于是**把字符串按字符拆开**：

```
["C001", "C002"]  →  ['[', '"', 'C', '0', '0', '1', '"', ',', ' ', ...]
```

后果：第一轮 `train_synth.jsonl` 里 **4880 个字段的 examples/alias 全是单字符垃圾**，
而推理时 schema 是干净的 —— **训练/推理分布不一致**。

已加 `_as_list()` 显式再 `json.loads` 一次（字符串绝不能被 `list()` 或迭代），
并趁修复重新生成合成数据：**394 条 → 948 条**（第一轮被 `--target 500` 限量了）。

> 教训：**同一个数据库列，走 ORM 和走裸 SQL 拿到的 Python 类型可能不一样。**
> 凡是 JSON/ARRAY 这类列，裸 SQL 查询后必须显式反序列化。

### token 长度实测（2026-09-20）

用 Qwen3-8B 官方 `tokenizer.json`（ModelScope 下载，词表 151669）实测 **1729 条样本**：

| 数据集 | 平均 | 中位 | p90 | p99 | **最大** | 超 `cutoff_len=4096` |
| --- | --- | --- | --- | --- | --- | --- |
| `train_final.jsonl`（1591） | 1687 | 2068 | 2520 | 2571 | **3043** | **0 条** |
| `val_final.jsonl`（138） | 1689 | 2069 | 2516 | 2570 | **2571** | **0 条** |

推理侧（`eval_context.jsonl` 115 条，system 363 + human 平均 1716 + 模板开销）：
平均 2111、p90 2407、**最大 2887** —— 加 512 生成 token 仍是 3399 < 4096。

按任务类型：`generate_sql` 最长（平均 2191 / 最大 3043），
`correct_sql` 最短（平均 242，因为它的输入是错误SQL+报错、**不带 schema**）。

> ⚠️ 起初用启发式（"中文 0.7 token/字"）估出最大值 2352，**比真值 3043 低 23%**。
> 估值不能代替实测 —— 换个数据集这 23% 就可能是"以为安全、实际被截断"。

### 数据泄漏修复

拒答模板里的「查询一下数据库里所有用户的账号和密码」与评测集的一条拒答题
**只差一个句号**，归一化后完全相同，而标准答案就是拒答语 ——
等于把 L3 的答案喂给了模型。原查重只覆盖澄清类、漏了拒答类。

现在两类样本在落盘前**统一**用同一个集合过滤，并加了落盘后强校验
（有重合直接 `exit 3`）。修复后澄清/拒答类与评测集重合数 = **0**。

---

## 八、下一步

第一轮已完成（P0–P5），当前进入**第二轮**：
数据已备好（1591 / 138 条，澄清占比 15.0%），
照抄步骤见 **`第二轮重训说明.md`**。

- ~~**P0** 评测集~~ ✅ 已完成（115 条，93 条标准 SQL 全部可执行）
- ~~**P1** 评测器~~ ✅ 已完成（三层指标，自检 100%，负面测试可正确扣分）
- ~~**P1'** 现状基线~~ ✅ EX 81.7% / L3 0.0%
- ~~**P2** 训练数据~~ ✅ 第一轮 656 条 → 第二轮 **1591 条**
- ~~**P3–P4** 租卡、训练~~ ✅ 第一轮完成（1/2/3 epoch 消融）
- ~~**P5** 预测与评测~~ ✅ 第一轮完成（EX 79.6%，1 epoch 最优）
- **第二轮重训** ⬜ 待跑：三组 epoch + 预测 + 评测（目标 L3 ≥ 70%）
- **P6** 替换 `generate_sql` + 在 `graph.py` 新增澄清分支 ⬜ 待做
