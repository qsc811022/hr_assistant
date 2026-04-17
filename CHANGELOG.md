# Changelog

## v3.0.0 — 2026-04-17

### 新增
- **`llm_client.py`**：全新多服務商 LLM 客戶端，支援 OpenAI、Anthropic (Claude)、Z.AI、Groq、Ollama 及任意 OpenAI 相容 API
- **AI Agent 架構**：LLM 透過 Tool Use 自主決定搜尋策略，可多次換關鍵字搜尋（最多 3 輪），再統整為最終回答
- **側欄 AI 設定面板**：服務商選單、API Key 輸入、模型名稱、Base URL，點擊套用即生效
- **`.streamlit/config.toml`**：關閉 Streamlit 檔案監看器，消除 torchvision 相關雜訊警告
- 自動從環境變數（`OPENAI_API_KEY`、`ANTHROPIC_API_KEY`、`GROQ_API_KEY`）帶入 API Key

### 改善
- **`rag_core.py`**：移除所有 LLM 程式碼，職責聚焦於文件載入與語義搜尋；修正 `normalize_embeddings=True` 確保 FAISS L2 距離正確
- **`app.py`**：改為使用者導向流程（無文件→上傳引導；有文件→問答介面）；移除 FAQ 快捷按鈕，改為通用文件問答 UI
- **`config.py`**：移除 LLM 相關設定（已移至 `llm_client.py`），精簡為純 RAG 參數
- **`requirements.txt`**：新增 `anthropic>=0.40.0`

### 修正
- 修正 FAISS 搜尋結果全被過濾的 bug：原因為 embedding 未正規化導致 L2 距離遠超 `MAX_DISTANCE=2.0` 閾值

---

## v2.0.0 — 2026-04-17

### 新增
- **`config.py`**：集中管理所有常數（模型名稱、chunk size、FAQ 清單、system prompt 等），消除散落在各檔案的魔術字串
- **`tests/test_rag_core.py`**：14 個單元測試，覆蓋文件載入、語義搜尋、相關度分數、fallback 行為
- **`docs/work_rules.txt`**：第二份範例政策文件，涵蓋工作規範、遠端工作、出差規定、績效考核
- **`README.md`**：完整專案文件，含系統架構圖、快速開始指南、參數說明表、技術棧

### 改善
- **`rag_core.py`**
  - 新增 `DocumentLoadError` 自訂例外，區分「找不到檔案」、「檔案過大」、「空白檔案」三種錯誤情境
  - 全函式型別提示（type hints）
  - 加入 Python logging，方便除錯與監控
  - API 呼叫加入指數退避重試機制（RateLimitError / ConnectionError）
  - `search()` 回傳結果新增 `relevance` 欄位（0–1 分數，由 FAISS L2 距離換算）
  - `ask_stream()` 支援帶入對話歷史（multi-turn context）
  - OpenAI client 與 SentenceTransformer 改為 lazy singleton，避免重複載入
- **`app.py`**
  - 空對話時顯示歡迎畫面
  - 文件解析與語義搜尋加入 Loading spinner
  - 低相關度（< 0.3）時主動顯示警告，引導聯絡 HR
  - 側欄新增 6 個 FAQ 快捷按鈕，一鍵提問
  - 對話匯出為 Markdown 檔案（下載按鈕）
  - 側欄文件清單顯示段落數與載入時間
  - 全面 try/except 錯誤處理，API 失敗時顯示友善訊息
  - 拆分為獨立函式（`_init_session`、`_render_sidebar`、`_handle_question` 等），提升可讀性
- **`requirements.txt`**：加入版本下限與 `pytest`
- **`.gitignore`**：補齊 `__pycache__`、`.pytest_cache`、`.venv`、`Thumbs.db` 等

## v1.0.0 — 初始版本

- 基礎 RAG 架構：Streamlit UI + FAISS 向量搜尋 + GLM-4.7 串流回覆
- 支援 `.txt` / `.pdf` 文件上傳
- 多語言語義搜尋（paraphrase-multilingual-MiniLM-L12-v2）
- 側欄文件管理與對話清除
