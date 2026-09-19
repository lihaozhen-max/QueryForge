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
| `gen_eval_set.py` | 生成评测集（模板抽样） | **仅标准库** |
| `validate_sql.py` | 校验标准 SQL 是否真能执行 | **仅标准库 + sqlite3** |
| `data/test.jsonl` | 评测集（每行一条样本） | 产物 |
| `data/test_stats.md` | 分层统计，可直接放进交付文档 | 产物 |

两个脚本**都只用 Python 标准库**，因此当前项目 `.venv` 不需要新增任何依赖，
用系统 Python 3.14 或 `.venv` 的 Python 3.13 都能直接跑。

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

## 五、下一步

已完成的只有 P0（评测集）。后续见 `微调规划.md` 第七节路线图：

- **P1** 跑现状基线，填对比矩阵第一行
- **P2** 写采集脚本，用现有 LangGraph 流水线产出 1000 条训练数据（复用 `compiled_graph`，不新增依赖）
- **P3–P4** 租卡、冒烟、QLoRA 训练
- **P5** 连真实 MySQL 跑执行准确率（EX）评测
- **P6** 替换 `generate_sql` + 在 `graph.py` 新增澄清分支
