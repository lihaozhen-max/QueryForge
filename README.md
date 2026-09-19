# QueryForge · 掌柜问数

> 一个基于 **LangGraph** 的中文自然语言转 SQL（Text2SQL / DB-RAG）智能问数 Agent。
> 用户用一句中文提问，系统自动检索数据仓库的元数据、生成并校验 SQL，最终返回数据库真实的查询结果。

例如输入「各个地区iPhone去年卖了多少钱」，系统会自己找到该用哪几张表、哪些字段、什么口径，写出 SQL 并执行。

---

## 一、它解决什么问题

传统 RAG（Doc-RAG）检索的是**文档片段**，让大模型读完之后直接"写答案"，因此容易编造数字。
本项目走的是 **DB-RAG** 路线：检索的是**数据库的元数据**（表、字段、指标口径、字段取值），让大模型只负责"把中文翻译成 SQL"，而**真正的计算与答案完全交给数据库执行**。

这样做的收益：

- 数字由数据库算出，**大模型无法编造数据**；
- 生成的 SQL 可以用数据库自身的解析器（`EXPLAIN`）校验，**语法错误可自动回环修正**；
- 输出是结构化结果集，天然适合渲染成表格 / 图表。

---

## 二、两个阶段

### 阶段一：离线构建元数据知识库

由人工维护的 `conf/meta_config.yaml`（数仓表 / 字段 / 指标的语义描述）驱动，把"数据的描述"写入三个存储：

```
conf/meta_config.yaml
   │
   ├─→ MySQL(meta)  表信息 / 字段信息 / 指标信息 / 字段-指标关联   结构化元数据
   ├─→ Qdrant       字段名+描述+别名、指标名+描述+别名 逐条向量化    语义召回
   └─→ ES           字段真实取值（sync:true 的字段）                字面/精确召回
```

一个字段会产生**多条向量**（name、description、每个 alias 各一条）但指向同一 payload，用别名提升召回率。

### 阶段二：在线问答（LangGraph 编排）

```
                          ┌─→ recall_column ─┐
START → extract_keywords ─┼─→ recall_metric ─┼─→ merge_retrieved_info
                          └─→ recall_value  ─┘          │
                                          ┌─────────────┴─────────────┐
                                     filter_metric               filter_table
                                          └─────────────┬─────────────┘
                                                  add_extra_context   ← 注入当前日期/季度 + 数据库版本
                                                        │
                                                   generate_sql
                                                        │
                                                   validate_sql ──通过──→ execute_sql → END
                                                        │
                                                    校验失败
                                                        ↓
                                                   correct_sql ──→ 回到 validate_sql 复验
                                                   （最多校正 2 次，超限则直接执行）
```

几个关键设计：

| 设计点 | 说明 |
| --- | --- |
| **三路混合召回** | 字段走**向量**（语义匹配）、取值走**全文索引**（字面匹配）、指标走**指标定义**（业务口径）。三者互补，覆盖"语义/字面/口径"三种召回需求 |
| **关键词双来源** | jieba 提词容易丢语义，因此把 **query 原句也作为一个 keyword** 加回去；每个召回节点再用大模型扩词，与 jieba 结果 `set()` 去重合并 |
| **合并召回不只是合并** | `merge_retrieved_info` 会把指标关联字段、召回取值对应的字段、以及每张表的**主键/外键**都补进 Schema，保证送给大模型的表结构自洽可 JOIN |
| **`EXPLAIN` 校验** | 用 `explain {sql}` 借数据库解析器发现"表不存在 / 字段错 / JOIN 语法错"，**不扫描数据**，比二次问大模型更廉价可靠 |
| **有上限的自愈环** | `validate_sql → correct_sql → validate_sql`，把 SQL 和报错回灌给大模型重写后**复验**；`MAX_CORRECT_ATTEMPTS` 限制轮数，避免死循环 |
| **两轮 LLM 过滤** | `filter_metric` / `filter_table` 把召回的宽集合裁成"回答该问题真正需要的表/字段"，控制上下文长度、降低幻觉 |

---

## 三、技术栈

| 类别 | 选型 |
| --- | --- |
| 语言 / 包管理 | Python 3.13 + [uv](https://docs.astral.sh/uv/) |
| Web 框架 | FastAPI + uvicorn，SSE 流式输出 |
| Agent 编排 | LangGraph `StateGraph`（State / Context / 节点 / 边） |
| LLM | DeepSeek（`langchain-deepseek`，经 LangChain `init_chat_model`） |
| Embedding | `langchain-huggingface` 调用 TEI 端点，`BAAI/bge-large-zh-v1.5`（1024 维） |
| 关系型数据库 | MySQL ×2：`dw`（业务数仓，SQL 实际执行处）、`meta`（元数据） |
| ORM / 驱动 | SQLAlchemy 2.0 **异步** + asyncmy |
| 向量数据库 | Qdrant（COSINE 距离） |
| 全文检索 | Elasticsearch 8（`ik_max_word` 中文分词） |
| 中文分词 | jieba |
| 配置 | OmegaConf（YAML → dataclass，支持环境变量注入） |
| 日志 | loguru（通过 `ContextVar` 注入 `request_id` 实现请求级链路追踪） |

---

## 四、目录结构

```
.
├── main.py                        # FastAPI 入口（中间件 + 路由 + lifespan）
├── conf/
│   ├── app_config.yaml            # 本地配置（含凭据，已被 .gitignore 忽略，需自行创建）
│   ├── app_config.example.yaml    # 配置模板，照此复制出 app_config.yaml
│   └── meta_config.yaml           # 数仓表/字段/指标的语义定义（构建知识库的输入）
├── prompt/                        # 8 个提示词文件，与代码解耦
└── app/
    ├── agent/                     # LangGraph 编排层
    │   ├── graph.py               # 建图：节点注册、边、条件边
    │   ├── state.py               # DataAgentState：图内流转的状态
    │   ├── context.py             # DataAgentContext：注入节点的依赖对象
    │   ├── llm.py                 # LLM 客户端
    │   └── nodes/                 # 12 个节点，一个文件一个节点
    ├── api/                       # 路由 / 依赖注入 / 请求体 Schema
    ├── clients/                   # MySQL / ES / Qdrant / Embedding 客户端管理器
    ├── repositories/              # 持久层：mysql(meta/dw) / es / qdrant
    ├── services/                  # 业务层：知识库构建、在线问答
    ├── models/                    # 数据模型（SQLAlchemy / TypedDict）
    ├── core/                      # lifespan / 日志 / request_id 上下文
    ├── conf/                      # 配置加载（OmegaConf → dataclass）
    └── scripts/
        └── build_meta_knowledge.py  # 阶段一入口：构建元数据知识库
```

---

## 五、快速开始

### 1. 准备依赖服务

需要提前启动：**MySQL**（建 `dw` 与 `meta` 两个库）、**Qdrant**、**Elasticsearch**（需安装 `ik` 中文分词插件）、**Embedding 服务**（TEI，映射 8081 端口）。

### 2. 安装依赖

```bash
uv sync
```

### 3. 配置

复制模板并按需修改：

```bash
cp conf/app_config.example.yaml conf/app_config.yaml
```

**凭据一律通过环境变量提供，不写进配置文件**（`app_config.yaml` 中的口令与 API Key 使用 OmegaConf 的 `${oc.env:...}` 占位）。

PowerShell：

```powershell
$env:DB_META_PASSWORD = "你的meta库口令"
$env:DB_DW_PASSWORD   = "你的dw库口令"
$env:LLM_API_KEY      = "你的DeepSeek API Key"
```

bash：

```bash
export DB_META_PASSWORD="你的meta库口令"
export DB_DW_PASSWORD="你的dw库口令"
export LLM_API_KEY="你的DeepSeek API Key"
```

可覆盖的变量：

| 变量 | 是否必填 | 默认值 |
| --- | --- | --- |
| `DB_META_PASSWORD` | **必填** | 无（缺失则启动即报错） |
| `DB_DW_PASSWORD` | **必填** | 无（缺失则启动即报错） |
| `LLM_API_KEY` | **必填** | 无（缺失则启动即报错） |
| `DB_META_HOST` / `DB_META_PORT` / `DB_META_DATABASE` | 可选 | `localhost` / `3306` / `meta` |
| `DB_DW_HOST` / `DB_DW_PORT` / `DB_DW_DATABASE` | 可选 | `localhost` / `3306` / `dw` |
| `DB_META_USER` / `DB_DW_USER` | 可选 | `root` |
| `LLM_MODEL_NAME` | 可选 | `deepseek-chat` |

### 4. 构建元数据知识库（阶段一，先跑一次）

```bash
python -m app.scripts.build_meta_knowledge
```

### 5. 启动服务（阶段二）

```bash
python main.py
```

服务监听 `0.0.0.0:8000`，接口为 `POST /api/query`，以 SSE 流式返回各节点进度与最终结果：

```bash
curl -N -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "各个地区iPhone去年卖了多少钱"}'
```

---

## 六、学习与复习材料

- `测试题/测试题.md` —— 33 道测试题详解，每题按「一句话结论 → 详细展开 → 项目落地（文件:行号）」组织；
- `测试题/测试题2.md` —— 纯净题目版，可自测；
- `问数_笔记.txt` —— 项目笔记。

---

## 七、说明

- `conf/app_config.yaml` 含数据库口令与 API Key，**已在 `.gitignore` 中忽略**，不会进入版本库；请从 `conf/app_config.example.yaml` 复制生成。
- `conf/meta_config.yaml` 只包含数仓表/字段/指标的语义描述，**不含任何凭据**，且是构建知识库的必要输入，因此正常纳入版本管理。
- 仓库中的代码部分带有教学性质的 `if __name__ == '__main__'` 自测块与 `app/api/test/` 示例模块，用于演示 FastAPI、SQLAlchemy、Qdrant、ES 等组件的独立用法，不影响主流程。
