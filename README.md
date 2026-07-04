# LangChain RAG 教程 - 个人知识库问答

一个基于 **LangChain** 构建的 RAG（检索增强生成）实战项目，支持**本地 Ollama**与**在线 DeepSeek**两种 LLM，帮助你快速入门 RAG 系统的核心概念和用法。

> 本项目展示 RAG 系统的完整实现：文档加载 → 向量索引 → 相似度检索 → LLM 生成答案

## 🎯 项目目标

- 理解 RAG（检索增强生成）的工作原理
- 掌握 LangChain 的核心概念（Document、Embeddings、VectorStore）
- 跑通个人知识库问答系统
- 了解向量索引、相似度检索等关键概念

## 🛠️ 技术栈

- **LangChain**：LLM 应用框架（核心组件 + LCEL）
- **langchain-ollama**：Ollama 官方集成（ChatOllama / OllamaEmbeddings）
- **langchain-deepseek**：DeepSeek 官方集成（ChatDeepSeek，可选，在线 LLM）
- **Ollama**：本地运行 LLM（deepseek-r1:1.5b）与 Embedding（nomic-embed-text）
- **FAISS**：向量数据库（稠密检索）
- **BM25（rank-bm25）**：倒排索引（稀疏检索）
- **jieba**：中文分词（让 BM25 在中文语料上真正生效）
- **python-dotenv**：从 .env 加载环境变量（安全管理 API Key）
- **RRF 融合算法**：多路检索结果合并
- **Python 3.11+**

## 📦 安装

```bash
# 1. 克隆项目
git clone <your-repo-url>
cd langchain-rag-tutorial

# 2. 同步依赖（推荐使用 uv）
uv sync

# 3. 确保 Ollama 运行中
ollama serve

# 4. 下载所需模型（如果未安装）
ollama pull deepseek-r1:1.5b
ollama pull nomic-embed-text

# 5.（可选）使用在线 DeepSeek 时，复制环境变量模板并填入 API Key
cp .env.example .env
#    然后编辑 .env，设置 DEEPSEEK_API_KEY 和 LLM_PROVIDER=deepseek
```

> **注意**：项目使用 `uv` 管理依赖，确保已安装 uv。如果未安装，请运行：`pip install uv`

## 🚀 运行

```bash
# 方式 1：使用 uv run（推荐）
uv run python main.py

# 方式 2：激活虚拟环境后运行
source .venv/bin/activate
python main.py

# 方式 3：直接使用虚拟环境的 Python
.venv/bin/python main.py
```

⚠️ **重要提示**：请使用 `uv run` 或激活虚拟环境后运行，否则可能遇到 `ModuleNotFoundError: No module named 'ollama'` 错误。

首次运行会自动从 `./data` 构建向量索引并持久化到 `./faiss_index/`；后续运行直接加载索引，速度快很多。

## 🌐 使用 DeepSeek 在线模型（可选）

本项目默认使用本地 Ollama（零成本、离线）。你也可以切换到**在线 DeepSeek**——价格极低（`deepseek-chat` 约 $0.14/百万输入 token）、质量更高（DeepSeek-V3.1），适合没有本地 GPU 或想要更好生成质量的场景。

### 本地 vs 在线

| 维度 | 本地 Ollama | 在线 DeepSeek |
|---|---|---|
| 成本 | 免费 | 极便宜（约 $0.14/1M 输入 token） |
| 质量 | r1:1.5b 较弱 | V3.1，接近 GPT-4 水平 |
| 硬件 | 需本地内存/显存 | 无要求 |
| 网络 | 离线 | 需联网 |
| 隐私 | 数据不出本机 | 数据上云 |

### 配置步骤

1. **获取 API Key**：访问 [DeepSeek 开放平台](https://platform.deepseek.com/api_keys) 创建并充值（[定价说明](https://api-docs.deepseek.com/quick_start/pricing)）。
2. **复制配置模板**：`cp .env.example .env`
3. **编辑 `.env`**，填入：
   ```env
   DEEPSEEK_API_KEY=sk-你的真实key
   LLM_PROVIDER=deepseek
   ```
4. **运行**（命令不变）：
   ```bash
   uv run python main.py
   ```

### 💡 学习点：在线 LLM + 本地 Embedding 的混合架构

DeepSeek **不提供 Embedding API**，因此本项目在在线模式下依然用本地 Ollama（`nomic-embed-text`）做嵌入，只有"生成答案"这一步走 DeepSeek。这演示了一个重要模式：

- **LLM 与 Embedding 解耦**：两者各取所长——LLM 用在线的高质量模型，Embedding 用本地零成本模型。
- **索引可复用**：因为 Embedding 模型不变，切换 LLM provider 时 `faiss_index/` 无需重建，可平滑迁移。
- **代码无感知**：`ChatOllama` 与 `ChatDeepSeek` 都实现 LangChain 的 `ChatModel` 接口，下游 `llm.invoke(prompt).content` 完全一致。

> 想看推理过程？在 `.env` 设置 `DEEPSEEK_MODEL=deepseek-reasoner`，答案前会打印模型的 `reasoning_content`（思考链）。

## 📖 知识库内容

`data/knowledge_base.txt` 包含以下主题：

- 什么是 RAG：定义、工作流程、解决的三大 LLM 痛点
- LangChain 核心组件：Document / TextSplitters / Embeddings / VectorStore / LLM / Retriever
- 稠密检索（FAISS）与稀疏检索（BM25）的原理与差异
- 混合检索与 RRF 倒数排名融合算法
- LCEL（LangChain 表达式语言）简介
- 应用场景与扩展方向

## 📁 项目结构

```
langchain-rag-tutorial/
├── README.md              # 本文件
├── main.py                # 主程序
├── pyproject.toml         # 依赖配置
├── .env.example           # 环境变量模板（复制为 .env 使用，.env 已被 gitignore）
├── data/
│   └── knowledge_base.txt # 知识库文件
├── faiss_index/           # 持久化的向量索引（运行时生成）
│   ├── index.faiss
│   └── index.pkl
└── tests/                 # 单元测试
    └── test_main.py
```

## 💡 核心代码流程

```
1. 加载文档 (load_documents)
       ↓
2. 文本分割 (CharacterTextSplitter)
       ↓
   ┌───────────────────────────────┐
   │  3a. 向量嵌入 → FAISS 稠密索引  │
   │  3b. BM25 倒排索引（稀疏）     │
   └───────────────────────────────┘
       ↓
4. 混合检索 (hybrid_search)
   → FAISS Top-5 + BM25 Top-5
   → RRF 倒数排名融合
       ↓
5. 拼 Prompt + LLM 生成答案 (ChatOllama.invoke)
```

## 🚀 扩展方向

1. **更多数据源**：PDF、Word、网页
2. **多文档管理**：支持多个知识库
3. **聊天历史**：支持多轮对话
4. **Web UI**：用 Gradio 或 Streamlit 做网页界面
5. **API 服务**：用 FastAPI 部署成 REST API

## 📚 相关链接

- [LangChain 文档](https://python.langchain.com)
- [LangChain GitHub](https://github.com/langchain-ai/langchain)
- [Ollama 文档](https://ollama.ai/docs)
- [FAISS 文档](https://faiss.ai)
- [DeepSeek API 文档](https://api-docs.deepseek.com)
- [DeepSeek 定价](https://api-docs.deepseek.com/quick_start/pricing)
- [RAG 原理介绍](https://python.langchain.com/docs/tutorials/rag)