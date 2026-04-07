"""
======================================================
LangChain RAG 示例：个人知识库问答
======================================================

本项目演示如何使用 LangChain + Ollama 构建 RAG 系统，
并实现 BM25 + FAISS 混合检索，用于回答基于个人知识库的问题。

依赖：
- langchain
- langchain-community
- langchain-text-splitters
- faiss-cpu
- ollama
- rank-bm25

使用方法：
1. 确保 Ollama 服务运行中
2. 已下载模型：deepseek-r1:1.5b, nomic-embed-text
3. 运行：python main.py

======================================================
"""

import os
import pickle
import ollama
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document as LCDocument
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import CharacterTextSplitter
from typing import List

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    print("⚠️ rank-bm25 未安装，BM25 检索将不可用。请运行: uv add rank-bm25")


# ============== 配置 ==============
LLM_MODEL = "deepseek-r1:1.5b"
EMBED_MODEL = "nomic-embed-text"
INDEX_DIR = "./faiss_index"
BM25_PKL = "./faiss_index/bm25_index.pkl"

# 检索参数
RETRIEVE_K = 5   # 每个检索器取多少条
FUISON_K = 5     # 混合检索融合时每个取多少条（最终送给 LLM 的）
RRF_K = 60       # RRF 融合参数，越大两种检索结果越均衡


# ============== Ollama 模型适配器 ==============

class OllamaChat:
    """Ollama 聊天模型包装类"""

    def __init__(self, model: str = "deepseek-r1:1.5b"):
        self.model = model

    def invoke(self, messages: List[dict]) -> str:
        """调用 LLM 生成回答"""
        if isinstance(messages, list) and len(messages) > 0:
            if hasattr(messages[0], "content"):
                content = (
                    messages[0].content
                    if len(messages) == 1
                    else "\n".join([m.content for m in messages])
                )
            else:
                content = (
                    messages[0] if isinstance(messages[0], str) else str(messages[0])
                )
        else:
            content = str(messages)

        response = ollama.chat(
            model=self.model, messages=[{"role": "user", "content": content}]
        )
        return response["message"]["content"]

    def __call__(self, prompt: str) -> str:
        return self.invoke([{"role": "user", "content": prompt}])


class OllamaEmbeddings(Embeddings):
    """Ollama Embedding 模型"""

    def __init__(self, model: str = "nomic-embed-text"):
        self.model_name = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return ollama.embeddings(model=self.model_name, prompt=text)["embedding"]


# ============== 初始化 ==============

def init_llm():
    print("🔧 初始化 Ollama LLM...")
    return OllamaChat(model=LLM_MODEL)


def init_embedding():
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
    # 简单分词：空格 + 小写化
    doc_texts = [doc.page_content for doc in documents]
    tokenized_corpus = [doc.split() for doc in doc_texts]
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

    Args:
        results: 多个检索器的结果列表，每项是 [{"doc": Document, "score": float}]
        k: 融合参数，越大越均衡（通常 60）
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
    """向量检索（稠密检索）"""
    docs = vectorstore.similarity_search(query, k=k)
    return [{"doc": doc, "score": 1.0 - i / (k + 1)} for i, doc in enumerate(docs)]


def sparse_search(bm25, doc_texts: List[str], query: str, documents: List[LCDocument], k: int) -> List[dict]:
    """BM25 检索（稀疏检索）"""
    if bm25 is None:
        return []

    tokenized_query = query.split()
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
        return [item["doc"] for item in dense_results[:FUISON_K]]

    # RRF 融合
    fused = reciprocal_rank_fusion([dense_results, sparse_results], k=RRF_K)
    fused_docs = [item["doc"] for item in fused[:FUISON_K]]

    # 打印融合后的排名详情（方便调试）
    print("   🏆 RRF 融合排名（Top-5）：")
    for i, item in enumerate(fused[:5], 1):
        preview = item["doc"].page_content[:40].replace("\n", " ")
        print(f"      [{i}] 分数={item['rrf_score']:.3f}  |  {preview}...")

    return fused_docs


# ============== 问答 ==============

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

    response = llm.invoke(prompt)
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

    # 测试问答（演示混合检索效果）
    questions = [
        "什么是 LlamaIndex？",
        "RAG 是什么？",
        "LlamaIndex 和 LangChain 有什么区别？",
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
