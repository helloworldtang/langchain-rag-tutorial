"""
======================================================
单元测试 - LlamaIndex 入门项目
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
        """测试数据目录存在"""
        assert os.path.exists("data"), "data 目录不存在"

    def test_knowledge_base_exists(self):
        """测试知识库文件存在"""
        assert os.path.exists("data/knowledge_base.txt"), "knowledge_base.txt 不存在"

    def test_knowledge_base_not_empty(self):
        """测试知识库文件不为空"""
        with open("data/knowledge_base.txt", "r", encoding="utf-8") as f:
            content = f.read()
        assert len(content) > 100, "知识库内容太少"

    def test_main_file_exists(self):
        """测试主程序文件存在"""
        assert os.path.exists("main.py"), "main.py 不存在"

    def test_pyproject_exists(self):
        """测试配置文件存在"""
        assert os.path.exists("pyproject.toml"), "pyproject.toml 不存在"


class TestKnowledgeBase:
    """测试知识库内容"""

    def test_contains_llamaindex_intro(self):
        """测试包含 LlamaIndex 介绍"""
        with open("data/knowledge_base.txt", "r", encoding="utf-8") as f:
            content = f.read()
        assert "LlamaIndex" in content, "知识库缺少 LlamaIndex 内容"

    def test_contains_rag_intro(self):
        """测试包含 RAG 介绍"""
        with open("data/knowledge_base.txt", "r", encoding="utf-8") as f:
            content = f.read()
        assert "RAG" in content, "知识库缺少 RAG 内容"

    def test_contains_concepts(self):
        """测试包含核心概念"""
        with open("data/knowledge_base.txt", "r", encoding="utf-8") as f:
            content = f.read()
        # 检查核心概念关键词
        required_keywords = ["文档加载器", "索引", "查询引擎", "检索"]
        for keyword in required_keywords:
            assert keyword in content, f"知识库缺少: {keyword}"


class TestMainCode:
    """测试主程序代码"""

    def test_import_ollama(self):
        """测试导入 ollama"""
        import ollama
        assert ollama is not None

    def test_import_faiss(self):
        """测试导入 FAISS"""
        from langchain_community.vectorstores import FAISS
        assert FAISS is not None

    def test_import_document(self):
        """测试导入 Document"""
        from langchain_core.documents import Document
        assert Document is not None

    def test_import_text_splitter(self):
        """测试导入文本分割器"""
        from langchain_text_splitters import CharacterTextSplitter
        assert CharacterTextSplitter is not None

    def test_model_config(self):
        """测试模型配置"""
        # 验证配置存在
        with open("main.py", "r", encoding="utf-8") as f:
            content = f.read()
        assert "deepseek-r1:1.5b" in content, "LLM 模型配置缺失"
        assert "nomic-embed-text" in content, "Embedding 模型配置缺失"


class TestFunctionality:
    """测试功能"""

    def test_document_loading(self):
        """测试文档加载功能"""
        from langchain_core.documents import Document
        from langchain_text_splitters import CharacterTextSplitter

        # 模拟加载
        with open("data/knowledge_base.txt", "r", encoding="utf-8") as f:
            content = f.read()

        # 分割
        text_splitter = CharacterTextSplitter(
            separator="\n\n",
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = text_splitter.split_text(content)

        # 验证
        assert len(chunks) > 0, "文档分割失败"
        assert len(chunks[0]) > 0, "分割内容为空"

    def test_document_creation(self):
        """测试 Document 对象创建"""
        from langchain_core.documents import Document

        doc = Document(
            page_content="测试内容",
            metadata={"source": "test.txt"}
        )

        assert doc.page_content == "测试内容"
        assert doc.metadata["source"] == "test.txt"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])