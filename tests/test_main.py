"""
======================================================
单元测试 - LangChain RAG 项目（含混合检索）
======================================================
"""

import pytest
import os
import sys

# 确保可以导入 main 模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestProjectStructure:
    """测试项目结构"""

    def test_data_directory_exists(self):
        assert os.path.exists("data"), "data 目录不存在"

    def test_knowledge_base_exists(self):
        assert os.path.exists("data/knowledge_base.txt"), "knowledge_base.txt 不存在"

    def test_knowledge_base_not_empty(self):
        with open("data/knowledge_base.txt", "r", encoding="utf-8") as f:
            content = f.read()
        assert len(content) > 100, "知识库内容太少"

    def test_main_file_exists(self):
        assert os.path.exists("main.py"), "main.py 不存在"

    def test_pyproject_exists(self):
        assert os.path.exists("pyproject.toml"), "pyproject.toml 不存在"


class TestKnowledgeBase:
    """测试知识库内容（LangChain RAG 主题）"""

    def test_contains_langchain_intro(self):
        with open("data/knowledge_base.txt", "r", encoding="utf-8") as f:
            content = f.read()
        assert "LangChain" in content, "知识库缺少 LangChain 介绍"

    def test_contains_rag_intro(self):
        with open("data/knowledge_base.txt", "r", encoding="utf-8") as f:
            content = f.read()
        assert "RAG" in content, "知识库缺少 RAG 介绍"

    def test_contains_concepts(self):
        """知识库应覆盖 LangChain 核心组件与混合检索概念"""
        with open("data/knowledge_base.txt", "r", encoding="utf-8") as f:
            content = f.read()
        required_keywords = ["Document", "Embeddings", "VectorStore", "混合检索", "RRF"]
        for keyword in required_keywords:
            assert keyword in content, f"知识库缺少: {keyword}"


class TestMainCode:
    """测试主程序代码"""

    def test_import_langchain_ollama(self):
        from langchain_ollama import ChatOllama, OllamaEmbeddings
        assert ChatOllama is not None
        assert OllamaEmbeddings is not None

    def test_import_faiss(self):
        from langchain_community.vectorstores import FAISS
        assert FAISS is not None

    def test_import_document(self):
        from langchain_core.documents import Document
        assert Document is not None

    def test_import_text_splitter(self):
        from langchain_text_splitters import CharacterTextSplitter
        assert CharacterTextSplitter is not None

    def test_model_config(self):
        with open("main.py", "r", encoding="utf-8") as f:
            content = f.read()
        assert "deepseek-r1:1.5b" in content, "LLM 模型配置缺失"
        assert "nomic-embed-text" in content, "Embedding 模型配置缺失"

    def test_bm25_dependency(self):
        """测试 rank-bm25 依赖已配置"""
        with open("pyproject.toml", "r", encoding="utf-8") as f:
            content = f.read()
        assert "rank-bm25" in content, "pyproject.toml 缺少 rank-bm25 依赖"

    def test_key_functions_exist(self):
        """测试核心函数存在"""
        import main
        for fn in [
            "reciprocal_rank_fusion",
            "hybrid_search",
            "dense_search",
            "sparse_search",
            "build_bm25_index",
            "tokenize",
            "clean_response",
        ]:
            assert hasattr(main, fn), f"缺少 {fn} 函数"


class TestFunctionality:
    """测试功能"""

    def test_document_loading(self):
        from langchain_core.documents import Document
        from langchain_text_splitters import CharacterTextSplitter

        with open("data/knowledge_base.txt", "r", encoding="utf-8") as f:
            content = f.read()

        text_splitter = CharacterTextSplitter(
            separator="\n\n", chunk_size=500, chunk_overlap=50
        )
        chunks = text_splitter.split_text(content)

        assert len(chunks) > 0, "文档分割失败"
        assert len(chunks[0]) > 0, "分割内容为空"

    def test_document_creation(self):
        from langchain_core.documents import Document

        doc = Document(page_content="测试内容", metadata={"source": "test.txt"})
        assert doc.page_content == "测试内容"
        assert doc.metadata["source"] == "test.txt"

    def test_tokenize_chinese(self):
        """中文必须被分词，不能整段当成一个 token（否则 BM25 失效）"""
        import main
        tokens = main.tokenize("什么是 RAG 检索增强生成")
        assert isinstance(tokens, list)
        assert len(tokens) > 1, "中文未被切分，BM25 在中文上会失效"
        # token 应已小写化、无空白
        assert all(isinstance(t, str) for t in tokens)
        assert all(t == t.strip().lower() and t for t in tokens)

    def test_tokenize_english(self):
        """英文按词切分"""
        import main
        tokens = main.tokenize("BM25 sparse retrieval")
        assert "bm25" in tokens
        assert "retrieval" in tokens

    def test_clean_response_strips_think(self):
        """清理 deepseek-r1 的 <think> 推理过程"""
        import main
        out = main.clean_response("<think>这是一段推理</think>正式回答")
        assert out == "正式回答"
        # 无 think 标签时原样返回
        assert main.clean_response("普通回答") == "普通回答"

    def test_rrf_fusion(self):
        """测试 RRF 融合算法"""
        import main
        from langchain_core.documents import Document

        doc_a = Document(page_content="文档A", metadata={"source": "a.txt"})
        doc_b = Document(page_content="文档B", metadata={"source": "b.txt"})
        doc_c = Document(page_content="文档C", metadata={"source": "c.txt"})

        # FAISS: A> B> C  ;  BM25: B> A> C
        results = [
            [{"doc": doc_a, "score": 1.0}, {"doc": doc_b, "score": 0.9}, {"doc": doc_c, "score": 0.5}],
            [{"doc": doc_b, "score": 1.0}, {"doc": doc_a, "score": 0.8}, {"doc": doc_c, "score": 0.4}],
        ]

        fused = main.reciprocal_rank_fusion(results, k=60)

        # A 和 B 各在不同检索器中排第1，RRF分数相近，但 A 在 FAISS 排第1贡献略大
        # C 在两路都排第3，RRF 分数最低
        assert len(fused) == 3, "融合结果数量不对"
        assert fused[0]["doc"].page_content in ["文档A", "文档B"], "Top-1 应该是 A 或 B"
        assert fused[2]["doc"].page_content == "文档C", "Top-3 应该是 C"
        assert all("rrf_score" in item for item in fused), "每项缺少 rrf_score"

    def test_rrf_no_duplicate_scores(self):
        """测试 RRF 同一文档在不同检索器中分数累加"""
        import main
        from langchain_core.documents import Document

        doc_a = Document(page_content="文档A", metadata={"source": "a.txt"})
        doc_b = Document(page_content="文档B", metadata={"source": "b.txt"})

        # 两路都返回相同文档 A（排第1）和 B（排第2）
        results = [
            [{"doc": doc_a, "score": 1.0}, {"doc": doc_b, "score": 0.9}],
            [{"doc": doc_a, "score": 1.0}, {"doc": doc_b, "score": 0.9}],
        ]

        fused = main.reciprocal_rank_fusion(results, k=60)
        score_a = next(item["rrf_score"] for item in fused if item["doc"].page_content == "文档A")
        score_b = next(item["rrf_score"] for item in fused if item["doc"].page_content == "文档B")

        # A 在两路都排第1，分数累加两次，应大于只排第1一次的分数
        assert score_a > score_b, "A 的 RRF 分数应大于 B"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
