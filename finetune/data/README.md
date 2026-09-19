# 微调数据流水线（finetune/data/）

本目录的数据全部由脚本生成，**不含人工标注**。每条数据都可通过"执行校验"或"机械反向变换"保证正确性。

## 数据流向

```
                        dw.sql（真实 DDL + 真实维度取值）
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        │                         │                         │
        ▼                         ▼                         ▼
  gen_eval_set.py          make_corpus.py            make_negatives.py
  （模板直接给标准 SQL）      （只给问题）          （由标准 SQL 机械反向生成）
        │                         │                         │
        ▼                         ▼                         ▼
   test.jsonl              corpus.txt                  negatives.jsonl
   115 条评测集            1035 条问题                  17 澄清 + 16 拒答
        │                         │                         │
        │                         ▼                         │
        │              build_train_set.py                   │
        │        （跑现有 LangGraph 流水线采 SQL             │
        │         + EXPLAIN/执行 双重校验）                  │
        │                         │                         │
        │                         ▼                         │
        │                   train_raw.jsonl                 │
        │                    （约 500-700 条）              │
        │                         │                         │
        └─────────────────────────┴─────────────────────────┘
                                  │
                          合并成最终训练集
                       train.jsonl（SQL + 澄清 + 拒答）
```

**关键分工**：训练集与评测集走**不同生成路径** ——
训练集是"流水线跑出来的 SQL"，评测集是"模板直接给出的标准 SQL"。
这样才不存在"用流水线自己的输出考自己"的退化。

## 文件清单

| 文件 | 是否入库 | 说明 |
| --- | --- | --- |
| `test.jsonl` | ✅ | 评测集 115 条（93 条标准 SQL + 22 条需澄清/拒答），**冻结，不参与训练** |
| `test_stats.md` | ✅ | 评测集分层统计 |
| `corpus.txt` | ✅ | 训练用问题语料 1035 条（只有问题，无 SQL） |
| `negatives.jsonl` | ✅ | 澄清 17 条 + 拒答 16 条 |
| `train.jsonl` | 运行后产生 | 最终训练集（SQL 生成样本 + 负样本） |
| `train_rejected.jsonl` | 运行后产生 | 被丢弃的样本及原因，**用于分析流水线短板** |

### 评测集样本字段

| 字段 | 含义 |
| --- | --- |
| `expected_type` | `SQL` = 模型应输出 SQL；`CLARIFY` = 模型应输出澄清或拒答 |
| `reference_sql` | 标准 SQL（仅 `expected_type=SQL` 时有值） |
| `refuse` | `true` = 危险/越权请求，**必须拒绝**；`false` = 信息不全，应提出澄清 |
| `missing` | 应澄清的要点（时间范围 / 统计粒度 / 指标口径） |
| `capability` | 对应的作业能力编号（见下） |

> `expected_type` 与 `reference_sql` 的一致性由生成脚本保证：**有标准 SQL 才标 SQL 类型**。
> 早期版本曾把危险请求标成 `SQL` 却没给标准 SQL，导致评测把正确拒答判为失败，
> 已由 `eval_ex.py --pred-from-ref` 自检发现并修正。

## 执行顺序

```bash
# ① 生成评测集（已跑通，无需重复）
python finetune/gen_eval_set.py
python finetune/validate_sql.py        # 校验 93/93 标准 SQL 可执行

# ② 生成训练问题语料
python finetune/make_corpus.py         # -> data/corpus.txt（1035 条）

# ③ 采集训练样本（**需要 MySQL/Qdrant/ES/Embedding 四个服务已启动**）
python finetune/build_train_set.py --limit 50     # 先冒烟
python finetune/build_train_set.py --resume       # 正式采集，可断点续跑

# ④ 生成负样本
python finetune/make_negatives.py      # -> data/negatives.jsonl

# ⑤ 自检评测器（不需要数据库服务）
python finetune/eval_ex.py --db sqlite --pred-from-ref    # 三层指标应全 100%
```

## 关于负样本的处理（重要）

**流水线产不出澄清样本** —— 现有 LangGraph 没有澄清能力，遇到信息不全的问题也会硬编 SQL。
而项目只有一个人、没有人工标注，无法手写上千条。

**解决方案：机械反向变换。**
标准 SQL 里必然带有时间条件（`d.month`/`d.quarter`）、分组维度（`GROUP BY`）和聚合口径（`SUM`/`AVG`/`COUNT`）。
把这些要素逐一摘掉，问题就自然变成"信息不全、应当反问"：

| 摘掉什么 | 对应必须澄清的要点 |
| --- | --- |
| 时间条件 | 时间范围 |
| 分组维度 | 统计粒度 |
| 聚合口径 | 指标口径 |

这个变换**可逆且按构造正确**，因此不需要人判断答案对不对。
拒答样本同理：把"查询"语义反向变成 DML/DDL 或越权读取，答案是标准拒答语。

> ⚠️ **注意**：反向变换必须用**自然口语模板 + 填入业务实体**来生成问题，
> 不能靠正则删词 —— 那样会产出"月的总体情况是多少"这种既不合语法、也不像真人提问的句子。
> 早期版本犯过这个错，已修正。

## 已知限制

- **澄清样本偏少（17 条）**。原因：反向变换要求源 SQL 含可摘除的时间/分组要素，
  而当前 SQL 池只有 93 条，命中率有限。
  缓解：澄清是**可泛化的行为**（学会"信息不足要反问"不依赖具体实体），
  训练集里配到 **50–100 条**即可起效；早期版本已能稳定学到。
  后续可从真实用户提问日志中补充。
- **训练集 SQL 样本数量取决于流水线表现**。被 `EXPLAIN` 或执行过滤掉的样本不会进入训练集，
  这是刻意的：宁可数据少，也不要错误标注。
  `train_rejected.jsonl` 会记录丢弃原因，可直接用于分析流水线短板。
- **数据时间范围只有 2025 Q1**。`corpus.txt` / `test.jsonl` 的生成模板已避开超范围的时间表述。

## 零新增依赖

以上四个脚本**只用 Python 标准库**（含 `sqlite3`）。
`build_train_set.py` 额外复用项目已有的运行时依赖（SQLAlchemy / langgraph 等），**不引入任何新库**，
因此 `pyproject.toml` 无需改动。
