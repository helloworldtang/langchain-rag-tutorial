# TODO — 功能扩展方向

> 现有代码层面的优化（索引内容指纹校验、中文友好切分、Prompt 规范化、
> 依赖版本约束、RRF chunk_id 去重等）已完成并测试通过。
> 以下是后续可选的**功能扩展**，按建议优先级粗略排列。

## 功能扩展

- [ ] **多文档源接入**：支持 PDF / Word / 网页等，替代当前单一 txt。
  - 技术路径：用 LangChain 的 `PyPDFLoader` / `Docx2txtLoader` / `WebBaseLoader`，
    将 `load_documents` 改为多源加载并统一成 `Document` 列表；metadata 补 `source` 类型。
- [ ] **重排序（Rerank）**：检索后用交叉编码器对 Top-K 精排，提升答案相关性。
  - 技术路径：接入 `bge-reranker`（本地 Ollama）或在线 rerank API，
    在 `hybrid_search` 的 RRF 融合后再加一道 rerank。
- [ ] **Web UI**：用 Gradio 或 Streamlit 做交互式问答界面，替代命令行演示。
- [ ] **多轮对话**：引入聊天历史（memory），支持上下文连贯的追问。
- [ ] **REST API**：用 FastAPI 把问答能力部署成 HTTP 服务。
- [ ] **多知识库管理**：支持多个知识库的加载与按主题路由检索。

## 备注

- 上述方向与 README「🚀 扩展方向」一致；本文件用于具体待办跟踪。
- 每个扩展完成后建议同步更新对应单测，保持 `uv run pytest` 全绿。
