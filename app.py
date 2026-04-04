"""
LlamaIndex Hello World 应用
将本地文档加载、索引，并使用 DeepSeek LLM 进行查询
"""
import os
import sys
from dotenv import load_dotenv
from typing import List

# 导入 LlamaIndex 核心模块
from llama_index.core import VectorStoreIndex, Document, SimpleDirectoryReader
from llama_index.core.settings import Settings

# 导入 DeepSeek 模块
from llama_index.llms.deepseek import DeepSeekLLM
from llama_index.embeddings.deepseek import DeepSeekEmbedding

# 加载环境变量
load_dotenv()

# 配置
DATA_DIR = "data"
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

if not DEEPSEEK_API_KEY:
    print("错误：未找到 DEEPSEEK_API_KEY 环境变量")
    print("请设置 DEEPSEEK_API_KEY 环境变量或在 .env 文件中添加 DEEPSEEK_API_KEY=sk-...")
    sys.exit(1)

def load_documents(data_dir: str) -> List[Document]:
    """
    加载文档
    
    Args:
        data_dir: 数据目录路径
    
    Returns:
        list: 文档列表
    """
    print(f"开始加载文档: {data_dir}")

    if not os.path.exists(data_dir):
        print(f"错误：数据目录 {data_dir} 不存在")
        return []

    # 使用 SimpleDirectoryReader 加载目录中的所有文本文件
    # 这会递归扫描目录
    reader = SimpleDirectoryReader(data_dir)
    documents = reader.load_data()

    print(f"成功加载 {len(documents)} 个文档")
    for idx, doc in enumerate(documents):
        print(f"  文档 {idx + 1}: {doc.metadata.get('file_name', '')}")
        print(f"    内容预览: {doc.text[:50]}...")
        print()

    return documents

def create_index(documents: List[Document], embed_model: DeepSeekEmbedding) -> VectorStoreIndex:
    """
    创建索引
    
    Args:
        documents: 文档列表
        embed_model: 嵌入模型
    
    Returns:
        VectorStoreIndex: 索引对象
    """
    print("开始创建索引")
    print(f"嵌入模型: {embed_model.model_name}")

    # 创建索引
    # VectorStoreIndex 是一个简单的索引类型，将向量存储在内存中
    # 这对于小型数据集（几百个文档）来说已经足够快了
    index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
    print("索引创建完成")

    return index

def query_documents(index: VectorStoreIndex, llm: DeepSeekLLM, query: str) -> str:
    """
    查询文档
    
    Args:
        index: 索引对象
        llm: 大语言模型
        query: 查询文本
    
    Returns:
        str: 回答文本
    """
    # 创建查询引擎
    # query_engine 会自动处理整个 RAG 流程：
    # 1. 将用户问题转换为向量（使用与索引相同的 embed_model）
    # 2. 在索引中找到最相关的文档片段
    # 3. 将查询和相关文档片段组织成一个 Prompt
    # 4. 将 Prompt 发送给 LLM
    # 5. 获取 LLM 的回答
    query_engine = index.as_query_engine(llm=llm)

    # 执行查询
    response = query_engine.query(query)

    return response

def main():
    """主函数"""
    print("=" * 60)
    print("  LlamaIndex Hello World")
    print("=" * 60)
    print()

    # 第一阶段：加载文档
    print("第一阶段：加载文档")
    print("-" * 60)

    documents = load_documents(DATA_DIR)

    if not documents:
        print("没有找到任何文档")
        return

    # 第二阶段：索引文档
    print("\n第二阶段：索引文档")
    print("-" * 60)

    # 初始化嵌入模型
    # 使用 DeepSeek 的嵌入模型
    # 这会将文本转换为向量，用于语义检索
    embed_model = DeepSeekEmbedding(
        model_name="deepseek-embedding",
        api_key=DEEPSEEK_API_KEY
    )

    # 创建索引
    index = create_index(documents, embed_model)

    # 第三阶段：查询
    print("\n第三阶段：查询")
    print("-" * 60)

    # 初始化 LLM
    # 使用 DeepSeek 的聊天模型
    # temperature=0 表示让模型回答更加确定、一致，减少"幻觉"
    llm = DeepSeekLLM(
        model="deepseek-chat",
        api_key=DEEPSEEK_API_KEY,
        temperature=0
    )

    print(f"大语言模型: {llm.model_name}")

    # 定义一些测试查询
    queries = [
        "Python 的设计哲学是什么？",
        "Python 在哪些领域很受欢迎？",
        "LlamaIndex 是什么？"
    ]

    # 执行查询
    for query in queries:
        print("-" * 60)
        print(f"用户问题: {query}")
        print("-" * 60)

        try:
            response = query_documents(index, llm, query)
            print(f"DeepSeek 回答: {response.response}")
        except Exception as e:
            print(f"查询失败: {e}")
        print()

    # 交互式查询
    print("-" * 60)
    print("进入交互模式 (输入 'exit' 退出)")
    print("-" * 60)
    print()

    while True:
        query = input("\n请输入你的问题: ")

        if query.lower() in ['exit', 'quit', '退出']:
            print("退出程序")
            break

        if not query:
            print("请输入有效的问题")
            continue

        print("-" * 60)
        print(f"用户问题: {query}")
        print("-" * 60)

        try:
            response = query_documents(index, llm, query)
            print(f"DeepSeek 回答: {response.response}")
        except Exception as e:
            print(f"查询失败: {e}")
        print()

if __name__ == "__main__":
    main()
