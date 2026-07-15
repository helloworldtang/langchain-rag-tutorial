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
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        assert RecursiveCharacterTextSplitter is not None

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
        """load_documents 用 RecursiveCharacterTextSplitter 切出非空块"""
        import main
        documents = main.load_documents()
        assert len(documents) > 0, "文档分割失败"
        assert all(doc.page_content.strip() for doc in documents), "存在空内容块"

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

    def test_clean_response_unclosed_think(self):
        """未闭合的 <think> 标签应被剥掉，不泄漏进答案"""
        import main
        # 成对：删除推理，保留答案
        assert main.clean_response("<think>推理</think>答案") == "答案"
        # 未闭合：剥掉标签，不残留 <think>
        out = main.clean_response("答案<think>未闭合推理")
        assert "<think>" not in out
        assert "</think>" not in out
        assert "答案" in out

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

    def test_rag_prompt_formats(self):
        """RAG_PROMPT 用 ChatPromptTemplate 渲染出 system + human 消息"""
        import main
        messages = main.RAG_PROMPT.invoke({"context": "上下文X", "question": "问题Y"})
        msgs = messages.to_messages()
        assert len(msgs) == 2
        assert msgs[0].type == "system"
        assert msgs[1].type == "human"
        assert "上下文X" in msgs[1].content
        assert "问题Y" in msgs[1].content

    def test_rrf_merges_by_chunk_id(self):
        """两路返回的不同实例、若 chunk_id 相同应被识别为同一文档并合并分数"""
        import main
        from langchain_core.documents import Document
        # 两个不同 Python 实例，但同一 chunk_id（模拟 FAISS 副本 vs 原始 documents）
        d1 = Document(page_content="内容A", metadata={"chunk_id": 0})
        d2 = Document(page_content="内容A", metadata={"chunk_id": 0})
        other = Document(page_content="内容B", metadata={"chunk_id": 1})
        assert main._doc_key(d1) == main._doc_key(d2) == 0
        results = [
            [{"doc": d1, "score": 1.0}, {"doc": other, "score": 0.5}],
            [{"doc": d2, "score": 1.0}, {"doc": other, "score": 0.4}],
        ]
        fused = main.reciprocal_rank_fusion(results, k=60)
        # chunk_id=0 在两路都排第1，累加两次 → Top-1；d1/d2 合并成 1 项
        assert fused[0]["doc"].metadata["chunk_id"] == 0
        assert len(fused) == 2


class TestProviderSwitch:
    """测试 LLM provider 切换逻辑（全程离线，不联网、不真实调用 API）"""

    def test_init_llm_ollama_returns_chatollama(self):
        """ollama provider 返回 ChatOllama"""
        import main
        from langchain_ollama import ChatOllama
        llm = main.init_llm("ollama")
        assert isinstance(llm, ChatOllama)

    def test_init_llm_default_is_ollama(self, monkeypatch):
        """不显式传 provider 时默认走 ollama（不破坏现有行为）"""
        import main
        from langchain_ollama import ChatOllama
        monkeypatch.setenv("LLM_PROVIDER", "ollama")
        llm = main.init_llm()
        assert isinstance(llm, ChatOllama)

    def test_init_llm_deepseek_missing_key_raises(self, monkeypatch):
        """切到 deepseek 但未配置 key → 友好报错（不真实联网）"""
        import main
        monkeypatch.setenv("LLM_PROVIDER", "deepseek")
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY"):
            main.init_llm("deepseek")

    def test_init_llm_deepseek_unavailable_raises(self, monkeypatch):
        """未安装 langchain-deepseek → 友好报错"""
        import main
        monkeypatch.setattr(main, "DEEPSEEK_AVAILABLE", False)
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-fake")  # 有 key 但库未装
        with pytest.raises(RuntimeError, match="langchain-deepseek"):
            main.init_llm("deepseek")

    def test_init_llm_deepseek_returns_chatdeepseek(self, monkeypatch):
        """deepseek + 有 key + 库可用 → 返回 ChatDeepSeek 实例（用假类避免联网）"""
        import main

        class _FakeChatDeepSeek:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        monkeypatch.setattr(main, "DEEPSEEK_AVAILABLE", True)
        monkeypatch.setattr(main, "ChatDeepSeek", _FakeChatDeepSeek)
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-fake")
        llm = main.init_llm("deepseek")
        assert isinstance(llm, _FakeChatDeepSeek)
        assert llm.kwargs["model"] == main.DEEPSEEK_MODEL
        assert llm.kwargs["api_key"] == "sk-fake"
        assert llm.kwargs["temperature"] == 0

    def test_init_embedding_always_ollama(self):
        """两种 provider 下 embedding 都用本地 Ollama（关键学习点）"""
        import main
        from langchain_ollama import OllamaEmbeddings
        emb = main.init_embedding()
        assert isinstance(emb, OllamaEmbeddings)

    def test_env_example_exists_and_is_template(self):
        """.env.example 存在且是模板（不含真实 key）"""
        assert os.path.exists(".env.example")
        with open(".env.example", encoding="utf-8") as f:
            content = f.read()
        assert "DEEPSEEK_API_KEY=sk-your-key-here" in content
        assert "LLM_PROVIDER=" in content

    def test_main_loads_dotenv(self):
        """main.py 必须 import 并调用 load_dotenv"""
        with open("main.py", encoding="utf-8") as f:
            content = f.read()
        assert "from dotenv import load_dotenv" in content
        assert "load_dotenv(" in content


class _FakeEmbeddings:
    """假 Embedding：按文本长度生成定维向量，避免测试时调用 Ollama。"""

    def __init__(self, dim=16):
        self.dim = dim

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)

    def _vec(self, text):
        return [((len(text) + i) % self.dim) / self.dim for i in range(self.dim)]


class TestIndexIntegrity:
    """测试索引内容指纹校验（修复：改了知识库会触发 FAISS 重建，不再用旧索引）"""

    def _setup(self, monkeypatch, tmp_path):
        """把索引目录与知识库路径重定向到临时目录，隔离真实 faiss_index/"""
        import main
        idx_dir = tmp_path / "idx"
        kb = tmp_path / "knowledge_base.txt"
        monkeypatch.setattr(main, "INDEX_DIR", str(idx_dir))
        monkeypatch.setattr(main, "INDEX_META_FILE", str(idx_dir / "index.meta.json"))
        monkeypatch.setattr(main, "DATA_FILE", kb)
        return main, idx_dir, kb

    def test_doc_signature_changes_with_content(self, tmp_path, monkeypatch):
        main, _, kb = self._setup(monkeypatch, tmp_path)
        kb.write_text("原文内容一", encoding="utf-8")
        sig1 = main._doc_signature()
        kb.write_text("原文内容一已被修改", encoding="utf-8")
        sig2 = main._doc_signature()
        assert sig1 != sig2, "内容变化后指纹应不同"

    def test_doc_signature_stable_when_unchanged(self, tmp_path, monkeypatch):
        main, _, kb = self._setup(monkeypatch, tmp_path)
        kb.write_text("稳定内容", encoding="utf-8")
        assert main._doc_signature() == main._doc_signature()

    def test_build_index_rebuilds_when_content_changes(self, tmp_path, monkeypatch, capsys):
        """改了知识库内容后，再次 build_index 应走重建分支"""
        main, _, kb = self._setup(monkeypatch, tmp_path)
        emb = _FakeEmbeddings()

        kb.write_text("段落一内容。\n\n段落二内容。\n\n段落三内容。", encoding="utf-8")
        main.build_index(main.load_documents(), emb)
        capsys.readouterr()  # 清空首次构建的输出

        kb.write_text(
            "段落一内容。\n\n段落二内容。\n\n段落三内容。\n\n段落四内容。\n\n段落五内容。",
            encoding="utf-8",
        )
        main.build_index(main.load_documents(), emb)
        out = capsys.readouterr().out
        assert "重建" in out, "内容变化后应提示重建索引"

    def test_build_index_reuses_when_unchanged(self, tmp_path, monkeypatch, capsys):
        """内容不变时再次 build_index 应复用已有索引（走加载分支）"""
        main, _, kb = self._setup(monkeypatch, tmp_path)
        emb = _FakeEmbeddings()
        kb.write_text("段落一内容。\n\n段落二内容。\n\n段落三内容。", encoding="utf-8")
        main.build_index(main.load_documents(), emb)  # 首次构建
        capsys.readouterr()

        main.build_index(main.load_documents(), emb)  # 指纹一致
        out = capsys.readouterr().out
        assert "加载" in out, "内容不变应复用已有索引"

    def test_bm25_not_persisted(self, tmp_path, monkeypatch):
        """BM25 不再落盘 pickle 文件（小语料每次重建）"""
        main, idx_dir, kb = self._setup(monkeypatch, tmp_path)
        kb.write_text("段落一内容。\n\n段落二内容。", encoding="utf-8")
        result = main.build_bm25_index(main.load_documents())
        assert result is not None, "rank-bm25 已安装时应返回索引"
        bm25, doc_texts = result
        assert len(doc_texts) == len(main.load_documents())
        assert not (idx_dir / "bm25_index.pkl").exists(), "BM25 不应再持久化"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
