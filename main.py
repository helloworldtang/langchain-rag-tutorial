"""
======================================================
LangChain RAG 示例：个人知识库问答（BM25 + FAISS 混合检索）
======================================================

本项目演示如何使用 LangChain + Ollama 构建 RAG 系统，
并实现 BM25 + FAISS 混合检索 + RRF 融合，用于回答基于个人知识库的问题。

技术点：
- LangChain 核心组件：Document / TextSplitters / Embeddings / VectorStore
- 官方 langchain-ollama 集成：ChatOllama（LLM）、OllamaEmbeddings（嵌入）
- 稠密检索（FAISS）+ 稀疏检索（BM25）+ RRF 倒数排名融合

依赖：
- langchain / langchain-community / langchain-text-splitters
- langchain-ollama（官方 Ollama 集成）
- faiss-cpu
- rank-bm25
- jieba（中文分词，可选）

使用方法：
1. 确保 Ollama 服务运行中（ollama serve）
2. 已下载模型：deepseek-r1:1.5b、nomic-embed-text
3. 运行：python main.py

======================================================
"""

import os
import pickle
import re
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document as LCDocument
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import CharacterTextSplitter

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


# ============== 配置 ==============
LLM_MODEL = "deepseek-r1:1.5b"
EMBED_MODEL = "nomic-embed-text"
INDEX_DIR = "./faiss_index"
BM25_PKL = "./faiss_index/bm25_index.pkl"

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

def init_llm():
    """初始化 LLM（通过官方 langchain-ollama 集成）"""
    print("🔧 初始化 Ollama LLM...")
    return ChatOllama(model=LLM_MODEL)


def init_embedding():
    """初始化 Embedding 模型（通过官方 langchain-ollama 集成）"""
    print("🔧 初始化 Ollama Embedding...")
    return OllamaEmbeddings(model=EMBED_MODEL)


# ============== 文档加载 ==============

def load_documents():
    """加载知识库文档"""
    print("📚 加载知识库文档...")

    with open("./data/knowledge_base.txt", "r", encoding="utf-8") as f:
        content = f.read()

    text_splitter = CharacterTextSplitter(
        separator="\n\n", chunk_size=500, chunk_overlap=50
    )
    chunks = text_splitter.split_text(content)

    documents = [
        LCDocument(page_content=chunk, metadata={"source": "knowledge_base.txt"})
        for chunk in chunks
        if chunk.strip()
    ]

    print(f"   已加载 {len(documents)} 个文档块")
    return documents


# ============== 索引构建 ==============

def build_bm25_index(documents: List[LCDocument]):
    """构建 BM25 倒排索引"""
    if not BM25_AVAILABLE:
        return None

    # 读取持久化索引（如果存在）
    if os.path.exists(BM25_PKL):
        print("📦 检测到已有 BM25 索引，加载中...")
        with open(BM25_PKL, "rb") as f:
            bm25_data = pickle.load(f)
        bm25 = bm25_data["bm25"]
        doc_texts = bm25_data["doc_texts"]
        return bm25, doc_texts

    print("🔍 构建 BM25 索引...")
    doc_texts = [doc.page_content for doc in documents]
    # 中文用 jieba/字符分词，英文同样适用
    tokenized_corpus = [tokenize(doc) for doc in doc_texts]
    bm25 = BM25Okapi(tokenized_corpus)

    # 持久化
    os.makedirs(INDEX_DIR, exist_ok=True)
    with open(BM25_PKL, "wb") as f:
        pickle.dump({"bm25": bm25, "doc_texts": doc_texts}, f)
    print("   BM25 索引创建完成！")
    return bm25, doc_texts


def build_index(documents, embed_model):
    """构建或加载 FAISS 向量索引"""
    if os.path.exists(INDEX_DIR) and os.path.exists(
        os.path.join(INDEX_DIR, "index.faiss")
    ):
        print("📦 检测到已有 FAISS 索引，加载中...")
        vectorstore = FAISS.load_local(
            INDEX_DIR, embed_model, allow_dangerous_deserialization=True
        )
        return vectorstore

    print("🔍 构建 FAISS 向量索引...")
    vectorstore = FAISS.from_documents(documents=documents, embedding=embed_model)
    print("💾 保存 FAISS 索引到本地...")
    vectorstore.save_local(INDEX_DIR)
    print("   FAISS 索引创建完成！")
    return vectorstore


# ============== 混合检索 ==============

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
            doc_key = item["doc"].page_content  # 用内容作为唯一 key
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

def clean_response(text: str) -> str:
    """
    清理 deepseek-r1 的输出：去掉 <think>...</think> 推理过程，只保留正式回答。
    若没有 think 标签则原样返回。
    """
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return cleaned or text


def answer_question(question: str, vectorstore, bm25, doc_texts, documents, llm) -> None:
    """
    回答用户问题（使用混合检索）
    """
    # 混合检索 Top-K 相关文档
    docs = hybrid_search(vectorstore, bm25, doc_texts, documents, question)

    # 构建上下文
    context = "\n\n".join([doc.page_content for doc in docs])

    # 构建 Prompt
    prompt = f"""基于以下上下文回答问题。如果无法从上下文找到答案，请如实说明。

上下文:
{context}

问题: {question}

回答:"""

    print(f"\n❓ 问题: {question}")
    print(f"📖 参考 {len(docs)} 个文档片段：")
    for i, doc in enumerate(docs, 1):
        preview = doc.page_content[:60].replace("\n", " ")
        print(f"   [{i}] {preview}...")

    # ChatOllama.invoke 返回 AIMessage，取 .content 得到字符串
    response = llm.invoke(prompt).content
    response = clean_response(response)
    print(f"\n✅ 答案:\n{response}")

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

    # 构建或加载 BM25 索引
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
    print("💬 开始混合检索问答测试（FAISS + BM25 + RRF）")
    print("=" * 55)

    for q in questions:
        answer_question(q, vectorstore, bm25, bm25_doc_texts, documents, llm)
        print("-" * 55)

    print("\n" + "=" * 55)
    print("✅ 测试完成！")
    print("=" * 55)


if __name__ == "__main__":
    main()
