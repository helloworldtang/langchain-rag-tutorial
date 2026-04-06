"""
======================================================
LlamaIndex 入门示例：个人知识库问答
======================================================

本项目演示如何使用 LangChain + Ollama 构建 RAG 系统
用于回答基于个人知识库的问题

依赖：
- langchain
- langchain-community
- langchain-text-splitters
- faiss-cpu
- ollama

使用方法：
1. 确保 Ollama 服务运行中
2. 已下载模型：deepseek-r1:1.5b, nomic-embed-text
3. 运行：python main.py
======================================================
"""

import os
import ollama
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document as LCDocument
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import CharacterTextSplitter
from typing import List


# ============== 配置 ==============
LLM_MODEL = "deepseek-r1:1.5b"
EMBED_MODEL = "nomic-embed-text"
INDEX_DIR = "./faiss_index"


class OllamaChat:
    """Ollama 聊天模型包装类"""

    def __init__(self, model: str = "deepseek-r1:1.5b"):
        self.model = model

    def invoke(self, messages: List[dict]) -> str:
        """调用 LLM 生成回答"""
        if isinstance(messages, list) and len(messages) > 0:
            # 如果是 ChatMessage 对象列表
            if hasattr(messages[0], 'content'):
                # 提取 content
                content = messages[0].content if len(messages) == 1 else "\n".join([m.content for m in messages])
            else:
                content = messages[0] if isinstance(messages[0], str) else str(messages[0])
        else:
            content = str(messages)
        
        response = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": content}]
        )
        return response["message"]["content"]

    def __call__(self, prompt: str) -> str:
        """直接调用"""
        return self.invoke([{"role": "user", "content": prompt}])


class OllamaEmbeddings(Embeddings):
    """Ollama Embedding 模型"""

    def __init__(self, model: str = "nomic-embed-text"):
        self.model_name = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return ollama.embeddings(model=self.model_name, prompt=text)["embedding"]


def init_llm():
    """初始化 LLM"""
    print("🔧 初始化 Ollama LLM...")
    return OllamaChat(model=LLM_MODEL)


def init_embedding():
    """初始化 Embedding 模型"""
    print("🔧 初始化 Ollama Embedding...")
    return OllamaEmbeddings(model=EMBED_MODEL)


def load_documents():
    """加载知识库文档"""
    print("📚 加载知识库文档...")
    
    with open("./data/knowledge_base.txt", "r", encoding="utf-8") as f:
        content = f.read()
    
    # 使用 CharacterTextSplitter 分割
    text_splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_text(content)
    
    # 转换为 LangChain Document
    documents = [
        LCDocument(page_content=chunk, metadata={"source": "knowledge_base.txt"})
        for chunk in chunks if chunk.strip()
    ]
    
    print(f"   已加载 {len(documents)} 个文档块")
    return documents


def build_index(documents, embed_model):
    """
    构建或加载索引
    """
    # 检查是否存在持久化索引
    if os.path.exists(INDEX_DIR) and os.path.exists(os.path.join(INDEX_DIR, "index.faiss")):
        print("📦 检测到已有索引，加载中...")
        vectorstore = FAISS.load_local(
            INDEX_DIR,
            embed_model,
            allow_dangerous_deserialization=True
        )
        return vectorstore

    # 构建 FAISS 向量索引
    print("🔍 构建向量索引...")
    vectorstore = FAISS.from_documents(
        documents=documents,
        embedding=embed_model
    )

    # 保存到本地
    print("💾 保存索引到本地...")
    vectorstore.save_local(INDEX_DIR)

    print("   索引创建完成！")
    return vectorstore


def answer_question(question: str, vectorstore, llm) -> None:
    """
    回答用户问题
    """
    # 检索相关文档
    docs = vectorstore.similarity_search(question, k=3)
    
    # 构建上下文
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # 构建 Prompt
    prompt = f"""基于以下上下文回答问题。如果无法从上下文找到答案，请如实说明。

上下文:
{context}

问题: {question}

回答:"""
    
    # 调用 LLM
    print(f"\n❓ 问题: {question}")
    response = llm.invoke(prompt)
    print(f"✅ 答案: {response}")

    # 展示引用来源
    if docs:
        print("📄 参考来源：")
        for i, doc in enumerate(docs, 1):
            filename = doc.metadata.get("source", "unknown")
            print(f"  [{i}] {filename}")


def main():
    """主函数"""
    # 初始化 LLM 和 Embedding
    llm = init_llm()
    embed_model = init_embedding()

    # 加载文档
    documents = load_documents()

    # 构建或加载索引
    vectorstore = build_index(documents, embed_model)

    # 测试问答
    questions = [
        "什么是 LlamaIndex？",
        "RAG 是什么？",
        "LlamaIndex 和 LangChain 有什么区别？",
    ]

    print("\n" + "=" * 50)
    print("💬 开始问答测试")
    print("=" * 50)

    for q in questions:
        answer_question(q, vectorstore, llm)

    print("\n" + "=" * 50)
    print("✅ 测试完成！")
    print("=" * 50)


if __name__ == "__main__":
    main()