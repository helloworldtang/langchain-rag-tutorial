# LangChain RAG 教程 - 个人知识库问答

一个基于 **LangChain + Ollama** 构建的 RAG（检索增强生成）实战项目，帮助你快速入门 RAG 系统的核心概念和用法。

> 本项目展示 RAG 系统的完整实现：文档加载 → 向量索引 → 相似度检索 → LLM 生成答案

## 🎯 项目目标

- 理解 RAG（检索增强生成）的工作原理
- 掌握 LangChain 的核心概念（Document、Embeddings、VectorStore）
- 跑通个人知识库问答系统
- 了解向量索引、相似度检索等关键概念

## 🛠️ 技术栈

- **LangChain**：LLM 应用框架
- **Ollama**：本地 LLM 运行（deepseek-r1:1.5b）
- **FAISS**：向量数据库
- **Python 3.11+**

## 📦 安装

```bash
# 1. 克隆项目
git clone <your-repo-url>
cd llamaindex-tutorials

# 2. 同步依赖（推荐使用 uv）
uv sync

# 3. 确保 Ollama 运行中
ollama serve

# 4. 下载所需模型（如果未安装）
ollama pull deepseek-r1:1.5b
ollama pull nomic-embed-text
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

## 📖 知识库内容

`data/knowledge_base.txt` 包含以下主题：

- RAG 系统简介与核心价值
- LangChain 核心概念（Document、Embeddings、VectorStore、LLM）
- RAG 原理与工作流程
- 向量数据库与相似度检索
- 应用场景与扩展方向

## 📁 项目结构

```
langchain-rag-tutorial/
├── README.md              # 本文件
├── main.py                # 主程序
├── pyproject.toml         # 依赖配置
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
3. 向量嵌入 (OllamaEmbeddings)
       ↓
4. 向量存储 (FAISS.from_documents)
       ↓
5. 相似度检索 (similarity_search)
       ↓
6. LLM 生成答案 (OllamaChat.invoke)
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
- [RAG 原理介绍](https://python.langchain.com/docs/tutorials/rag)