"""
======================================================
LangChain RAG 示例：个人知识库问答（BM25 + FAISS 混合检索）
======================================================

本项目演示如何使用 LangChain 构建 RAG 系统，
并实现 BM25 + FAISS 混合检索 + RRF 融合，用于回答基于个人知识库的问题。

技术点：
- LangChain 核心组件：Document / TextSplitters / Embeddings / VectorStore
- LLM provider 切换：本地 Ollama（默认）或 在线 DeepSeek（langchain-deepseek）
- 在线 LLM（DeepSeek）+ 本地 Embedding（Ollama）的混合架构
- 稠密检索（FAISS）+ 稀疏检索（BM25）+ RRF 倒数排名融合

依赖：
- langchain / langchain-community / langchain-text-splitters
- langchain-ollama（Ollama 官方 LLM/Embedding 集成）
- langchain-deepseek（DeepSeek 官方 LLM 集成，可选）
- faiss-cpu / rank-bm25 / jieba / python-dotenv

使用方法：
1. （默认）本地 Ollama：确保 ollama serve 运行，已下载 deepseek-r1:1.5b、nomic-embed-text
2. （可选）在线 DeepSeek：在 .env 设置 DEEPSEEK_API_KEY 和 LLM_PROVIDER=deepseek
3. 运行：python main.py

======================================================
"""

import hashlib
import json
import os
import re
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document as LCDocument
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 尽早加载 .env（必须先于下方使用 os.getenv 的配置区）
load_dotenv(Path(__file__).resolve().parent / ".env")

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    print("⚠️ rank-bm25 未安装，BM25 检索将不可用。请运行: uv add rank-bm25")

try:
    import jieba
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False
    print("⚠️ jieba 未安装，中文将按字符切分（效果较差）。建议运行: uv add jieba")

try:
    from langchain_deepseek import ChatDeepSeek
    DEEPSEEK_AVAILABLE = True
except ImportError:
    ChatDeepSeek = None  # 赋 None 兜底，避免未安装时引用未定义名
    DEEPSEEK_AVAILABLE = False
    print("⚠️ langchain-deepseek 未安装，DeepSeek 在线模型不可用。请运行: uv add langchain-deepseek")


# ============== 配置 ==============
# LLM provider：ollama（本地，默认）| deepseek（在线，需 DEEPSEEK_API_KEY）
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()
# 本地 Ollama 模型
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "deepseek-r1:1.5b")
# 在线 DeepSeek 模型：deepseek-chat（V3.1，通用）| deepseek-reasoner（推理型，返回 reasoning_content）
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

# Embedding：两种 provider 下都用本地 Ollama（DeepSeek 不提供 Embedding API）
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")

# 知识库源文件（提为常量，便于复用与测试 monkeypatch）
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "knowledge_base.txt"

# 索引目录与元数据文件（FAISS 旁路存放一份内容指纹，用于判断是否需要重建）
INDEX_DIR = str(BASE_DIR / "faiss_index")
INDEX_META_FILE = os.path.join(INDEX_DIR, "index.meta.json")

# 文本切分参数
# 中文信息密度高，300 字符/块比默认 500 更合适；分隔符按「段落→行→中文句末标点→逗号→字」逐级递归，
# 尽量在自然语义边界切分，避免把句子从中间切断。
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
CHINESE_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]

# 检索参数
RETRIEVE_K = 5   # 每个检索器取多少条
FUSION_K = 5     # 融合后最终送给 LLM 的文档数
RRF_K = 60       # RRF 融合参数，越大两种检索结果越均衡


# ============== 中文分词 ==============

def tokenize(text: str) -> List[str]:
    """
    文本分词，用于 BM25 倒排索引。

    - 优先使用 jieba 做中文分词（效果好）
    - 未安装 jieba 时，退化为按字符切分（对中文仍比 str.split() 强）
    - 统一小写，过滤空白 token

    注意：中文 BM25 必须分词，直接用空格 split 对中文几乎无效。
    """
    if JIEBA_AVAILABLE:
        tokens = list(jieba.cut(text))
    else:
        # 退化方案：按字符切，去掉空白字符
        tokens = [ch for ch in text if not ch.isspace()]
    return [t.strip().lower() for t in tokens if t.strip()]


# ============== 初始化 ==============

def init_llm(provider: str | None = None):
    """
    初始化 LLM（返回统一的 LangChain ChatModel，下游代码无感知具体实现）。

    - ollama（默认）：本地 ChatOllama，零成本、离线
    - deepseek：在线 ChatDeepSeek，质量高、便宜，需 DEEPSEEK_API_KEY

    注意：embedding 在两种 provider 下都用本地 Ollama（见 init_embedding），
    因为 DeepSeek 不提供 Embedding API——这正是"在线 LLM + 本地 Embedding"
    混合架构的典型做法：LLM 用在线的高质量模型，Embedding 用本地零成本模型。
    """
    # 实时读取环境变量作为默认值（便于测试用 monkeypatch 切换，无需重启进程）
    provider = (provider or os.getenv("LLM_PROVIDER", "ollama")).lower()
    if provider == "deepseek":
        if not DEEPSEEK_AVAILABLE:
            raise RuntimeError(
                "未安装 langchain-deepseek，无法使用 DeepSeek。请运行: uv add langchain-deepseek"
            )
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError(
                "未配置 DEEPSEEK_API_KEY。请复制 .env.example 为 .env 并填入 API Key"
                "（获取地址：https://platform.deepseek.com/api_keys）"
            )
        print(f"🔧 初始化 DeepSeek LLM (model={DEEPSEEK_MODEL})...")
        # temperature=0 让 RAG 答案稳定可复现
        return ChatDeepSeek(model=DEEPSEEK_MODEL, api_key=api_key, temperature=0)

    # 默认 ollama
    print(f"🔧 初始化 Ollama LLM (model={OLLAMA_LLM_MODEL})...")
    return ChatOllama(model=OLLAMA_LLM_MODEL, temperature=0)


def init_embedding():
    """
    初始化 Embedding 模型（本地 Ollama）。

    无论 LLM provider 是 ollama 还是 deepseek，embedding 都用本地 Ollama：
    - DeepSeek 不提供 Embedding API
    - 本地 embedding 零成本，向量索引可与在线 LLM 解耦复用（切换 LLM 无需重建索引）
    """
    print("🔧 初始化 Ollama Embedding...")
    return OllamaEmbeddings(model=EMBED_MODEL)


# ============== 文档加载 ==============

def load_documents():
    """加载知识库文档，按中文友好的分隔符递归切分成语义块。"""
    print("📚 加载知识库文档...")

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        content = f.read()

    # RecursiveCharacterTextSplitter 会按 separators 列表逐级尝试切分：
    # 优先在段落（\n\n）边界切，块仍超长再退到行、再到中文句末标点（。！？；），
    # 尽量保留完整句子，对中文语料比单分隔符的 CharacterTextSplitter 友好得多。
    text_splitter = RecursiveCharacterTextSplitter(
        separators=CHINESE_SEPARATORS,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = text_splitter.split_text(content)

    # 给每个块分配稳定 chunk_id，供 RRF 跨检索器去重（见 _doc_key）。
    clean_chunks = [c for c in chunks if c.strip()]
    documents = [
        LCDocument(
            page_content=chunk,
            metadata={"source": "knowledge_base.txt", "chunk_id": idx},
        )
        for idx, chunk in enumerate(clean_chunks)
    ]

    print(f"   已加载 {len(documents)} 个文档块")
    return documents


# ============== 索引指纹 ==============

def _doc_signature() -> dict:
    """
    计算当前知识库的「指纹」：源文档内容 hash + 切分配置 + embedding 模型。

    只要其中任一发生变化（改了知识库正文、调整 chunk_size、换了 embedding 模型），
    指纹就会不同，从而触发 FAISS 索引重建——避免「改了知识库却仍检索旧索引」的隐患。
    """
    content = Path(DATA_FILE).read_text(encoding="utf-8")
    return {
        "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "chunk_config": {"chunk_size": CHUNK_SIZE, "chunk_overlap": CHUNK_OVERLAP},
        "embed_model": EMBED_MODEL,
    }


def _load_index_meta() -> dict | None:
    """读取已持久化的索引指纹；不存在或损坏时返回 None。"""
    if not os.path.exists(INDEX_META_FILE):
        return None
    try:
        with open(INDEX_META_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _save_index_meta(meta: dict) -> None:
    """写入索引指纹。"""
    os.makedirs(INDEX_DIR, exist_ok=True)
    with open(INDEX_META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


# ============== 索引构建 ==============

def build_bm25_index(documents: List[LCDocument]):
    """
    构建 BM25 倒排索引（每次启动重建，不做持久化）。

    BM25 在小语料上构建是纯 CPU、毫秒级操作，无需像 FAISS 那样持久化——
    持久化反而会带来 pickle 反序列化的脆弱性（依赖版本变动后可能加载失败）
    与内容不同步风险。每次重建更简单、更可靠。
    """
    if not BM25_AVAILABLE:
        return None

    print("🔍 构建 BM25 索引...")
    doc_texts = [doc.page_content for doc in documents]
    # 中文用 jieba/字符分词，英文同样适用
    tokenized_corpus = [tokenize(doc) for doc in doc_texts]
    bm25 = BM25Okapi(tokenized_corpus)
    print("   BM25 索引构建完成！")
    return bm25, doc_texts


def build_index(documents, embed_model):
    """
    构建或加载 FAISS 向量索引。

    与「仅判断文件是否存在」不同，这里额外校验索引指纹是否匹配当前知识库：
    - 索引存在且指纹一致 → 直接加载（embedding 计算耗时，复用索引能省时间）
    - 索引不存在，或知识库/切分配置/embedding 模型已变更 → 重建
    """
    sig = _doc_signature()
    faiss_ok = os.path.exists(os.path.join(INDEX_DIR, "index.faiss"))
    meta_ok = _load_index_meta() == sig

    if faiss_ok and meta_ok:
        print("📦 检测到有效 FAISS 索引（指纹匹配），加载中...")
        return FAISS.load_local(
            INDEX_DIR, embed_model, allow_dangerous_deserialization=True
        )

    if faiss_ok:
        print("♻️ 知识库或切分配置已变更，重建 FAISS 索引...")
    else:
        print("🔍 构建 FAISS 向量索引...")

    vectorstore = FAISS.from_documents(documents=documents, embedding=embed_model)
    print("💾 保存 FAISS 索引到本地...")
    vectorstore.save_local(INDEX_DIR)
    _save_index_meta(sig)
    print("   FAISS 索引创建完成！")
    return vectorstore


# ============== 混合检索 ==============

def _doc_key(doc: LCDocument):
    """文档去重键：优先用稳定的 chunk_id，回退到 page_content。

    不能用 id(doc)：稠密检索（FAISS 存的是副本）与稀疏检索（原始 documents 列表）
    返回的是不同 Python 实例，对象 id 不同，会导致同一文档无法合并、破坏 RRF 融合。
    用 chunk_id 既能跨检索器正确合并，又不受「两块内容巧合相同」的误合并影响。
    """
    cid = doc.metadata.get("chunk_id")
    return cid if cid is not None else doc.page_content


def reciprocal_rank_fusion(results: List[List[dict]], k: int = 60) -> List[dict]:
    """
    RRF（倒数排名融合）算法

    融合多个检索器的结果，输出综合排名的文档列表。
    对每个检索结果中的文档，根据其排名分配分数：
      score = 1 / (k + rank)
    排名越靠前，分数越高，最后按总分排序。

    RRF 只用"排名"而非原始分数，因为不同检索器（向量相似度 vs BM25）
    的分数量纲完全不同，直接相加没有意义。

    Args:
        results: 多个检索器的结果列表，每项是 [{"doc": Document, "score": float}]
        k: 融合参数，越大越均衡（通常取 60）
    Returns:
        按 RRF 综合分数排序的文档列表
    """
    doc_scores = {}

    for retriever_results in results:
        for rank, item in enumerate(retriever_results, start=1):
            doc_key = _doc_key(item["doc"])
            if doc_key not in doc_scores:
                doc_scores[doc_key] = {"doc": item["doc"], "rrf_score": 0}
            # 累加 RRF 分数
            doc_scores[doc_key]["rrf_score"] += 1 / (k + rank)

    # 按 RRF 分数降序排列
    fused = sorted(doc_scores.values(), key=lambda x: x["rrf_score"], reverse=True)
    return fused


def dense_search(vectorstore, query: str, k: int) -> List[dict]:
    """
    向量检索（稠密检索）。

    使用 similarity_search_with_score 拿到真实相似度分数。
    FAISS 默认返回 L2 距离，越小越相似；返回结果已按距离升序排列（最相似在前）。
    注意：RRF 只使用排名，不使用这里的 score；score 仅用于展示与调试。
    """
    docs_with_scores = vectorstore.similarity_search_with_score(query, k=k)
    return [{"doc": doc, "score": float(score)} for doc, score in docs_with_scores]


def sparse_search(bm25, doc_texts: List[str], query: str, documents: List[LCDocument], k: int) -> List[dict]:
    """BM25 检索（稀疏检索）"""
    if bm25 is None:
        return []

    tokenized_query = tokenize(query)
    scores = bm25.get_scores(tokenized_query)

    # 取 Top-K，返回 (doc, score) 对
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [
        {"doc": documents[idx], "score": float(scores[idx])}
        for idx in top_indices
        if scores[idx] > 0
    ]


def hybrid_search(vectorstore, bm25, doc_texts, documents, query: str) -> List[LCDocument]:
    """
    混合检索：BM25（稀疏）+ FAISS（稠密）+ RRF 融合

    混合检索的好处：
    - BM25 擅长关键词匹配（专有名词、精确短语）
    - FAISS 擅长语义相似（同义词、语义关联）
    - 两者结合，覆盖更全面的检索场景
    """
    print(f"   🔎 混合检索: query =「{query}」")

    # 两路并行检索
    dense_results = dense_search(vectorstore, query, k=RETRIEVE_K)

    if bm25 is not None:
        sparse_results = sparse_search(bm25, doc_texts, query, documents, k=RETRIEVE_K)
    else:
        sparse_results = []

    # 展示两路检索各自的结果
    print(f"   📊 FAISS 命中: {len(dense_results)} 条 | BM25 命中: {len(sparse_results)} 条")

    if not sparse_results:
        # 无 BM25 时，直接用向量检索结果
        return [item["doc"] for item in dense_results[:FUSION_K]]

    # RRF 融合
    fused = reciprocal_rank_fusion([dense_results, sparse_results], k=RRF_K)
    fused_docs = [item["doc"] for item in fused[:FUSION_K]]

    # 打印融合后的排名详情（方便调试）
    print("   🏆 RRF 融合排名（Top-5）：")
    for i, item in enumerate(fused[:5], 1):
        preview = item["doc"].page_content[:40].replace("\n", " ")
        print(f"      [{i}] 分数={item['rrf_score']:.3f}  |  {preview}...")

    return fused_docs


# ============== 问答 ==============

# RAG 问答 Prompt 模板：system 约束"仅基于上下文回答"，human 放上下文与问题。
# 用 ChatPromptTemplate 而非裸字符串，规范 system/human 角色，便于复用与切换模型。
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "你是一个严谨的问答助手。请只基于下面提供的上下文回答问题；"
               "如果上下文中没有相关内容，请如实说明，不要编造。"),
    ("human", "上下文：\n{context}\n\n问题：{question}"),
])


def clean_response(text: str) -> str:
    """
    清理本地 deepseek-r1 的输出：去掉 <think>...</think> 推理过程，只保留正式回答。
    在线 DeepSeek（deepseek-chat / deepseek-reasoner）的 .content 本身就是干净答案，
    此函数对在线模型是无害的 no-op（匹配不到 <think> 时原样返回）。
    """
    # 1. 去掉成对的 <think>...</think> 推理过程（本地 deepseek-r1）
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # 2. 兜底：偶发的未闭合 <think>——剥掉残留的标签本身，避免泄漏进答案；
    #    不做激进截断，防止误删可能存在的正式答案。
    cleaned = cleaned.replace("<think>", "").replace("</think>", "")
    cleaned = cleaned.strip()
    return cleaned or text


def answer_question(question: str, vectorstore, bm25, doc_texts, documents, llm) -> None:
    """
    回答用户问题（使用混合检索）
    """
    # 混合检索 Top-K 相关文档
    docs = hybrid_search(vectorstore, bm25, doc_texts, documents, question)

    # 构建上下文
    context = "\n\n".join([doc.page_content for doc in docs])

    print(f"\n❓ 问题: {question}")
    print(f"📖 参考 {len(docs)} 个文档片段：")
    for i, doc in enumerate(docs, 1):
        preview = doc.page_content[:60].replace("\n", " ")
        print(f"   [{i}] {preview}...")

    # 用 ChatPromptTemplate 渲染 system + human 消息（规范的 prompt，便于复用与切换模型）
    messages = RAG_PROMPT.invoke({"context": context, "question": question})
    # LLM.invoke 接收消息，返回 AIMessage（本地 ChatOllama 与在线 ChatDeepSeek 均适用），取 .content 得到字符串
    response = llm.invoke(messages)
    # 在线 deepseek-reasoner 会把推理放进 additional_kwargs["reasoning_content"]，可选展示
    reasoning = response.additional_kwargs.get("reasoning_content") if response.additional_kwargs else None
    if reasoning:
        print(f"🧠 (推理过程) {reasoning[:200]}...")
    answer = clean_response(response.content)
    print(f"\n✅ 答案:\n{answer}")

    if docs:
        print("📄 参考来源：")
        for i, doc in enumerate(docs, 1):
            print(f"   [{i}] {doc.metadata.get('source', 'unknown')}")


# ============== 主函数 ==============

def main():
    llm = init_llm()
    embed_model = init_embedding()

    documents = load_documents()

    # 构建或加载 FAISS 索引
    vectorstore = build_index(documents, embed_model)

    # 构建 BM25 索引（每次重建，无需持久化）
    bm25_data = build_bm25_index(documents)
    bm25 = bm25_data[0] if bm25_data else None
    bm25_doc_texts = bm25_data[1] if bm25_data else []

    # 测试问答（演示混合检索效果，问题均能从知识库找到答案）
    questions = [
        "什么是 RAG？它解决了什么问题？",
        "LangChain 有哪些核心组件？",
        "FAISS 稠密检索和 BM25 稀疏检索有什么区别？",
    ]

    print("\n" + "=" * 55)
    print(f"💬 开始混合检索问答测试（provider={LLM_PROVIDER}）")
    print("=" * 55)

    for q in questions:
        answer_question(q, vectorstore, bm25, bm25_doc_texts, documents, llm)
        print("-" * 55)

    print("\n" + "=" * 55)
    print("✅ 测试完成！")
    print("=" * 55)


if __name__ == "__main__":
    main()
