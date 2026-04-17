# Changelog

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
