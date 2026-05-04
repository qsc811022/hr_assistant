# HR Assistant - RAG 文件問答助理

HR Assistant 是一個以 Streamlit 製作的文件問答系統。使用者可以上傳公司規章、HR 政策或 FAQ 文件，系統會將文件切片、轉成向量、建立 FAISS 索引，並透過 LLM agent 查詢相關段落後回答問題。

這個專案適合放在個人作品集的 `side_project:g4`，展示 RAG、向量搜尋、LLM tool use、多模型 API 串接與 Streamlit 產品化能力。

## Demo 流程

1. 在側邊欄選擇 LLM provider。
2. 輸入 API key、模型名稱與 base URL。
3. 上傳 `.txt` 或 `.pdf` 文件。
4. 在聊天框詢問 HR 或文件相關問題。
5. 系統會搜尋文件片段，交給 LLM 產生回答，並顯示引用來源。
6. 可將問答紀錄匯出成 Markdown。

## 主要功能

- 文件上傳：支援 `.txt`、`.pdf`。
- RAG 檢索：使用 `sentence-transformers` 產生 embedding，使用 FAISS 做相似度搜尋。
- Agent 查詢：LLM 可透過 `search_documents` tool 自主查詢文件內容。
- 多 LLM provider：支援 Z.AI、OpenAI、Anthropic、Groq、Ollama，以及 OpenAI-compatible API。
- 串流回覆：使用 Streamlit 即時顯示 LLM 回答。
- 問答匯出：可下載 Markdown 格式的 Q&A 紀錄。
- 基礎測試：`tests/test_rag_core.py` 驗證文件載入與搜尋邏輯。

## 技術架構

```text
Streamlit UI (app.py)
        |
        v
LLM Agent (llm_client.py)
        |
        | tool call: search_documents(query)
        v
RAG Core (rag_core.py)
        |
        | text chunking + embedding
        v
FAISS Vector Index
        |
        v
Uploaded HR documents / policy files
```

## 專案結構

```text
hr_assistant/
├── app.py                 # Streamlit UI 與使用者互動流程
├── rag_core.py            # 文件讀取、切片、embedding、FAISS 搜尋
├── llm_client.py          # 多 provider LLM client 與 agent tool-use loop
├── config.py              # RAG 與 app 設定
├── requirements.txt       # Python 套件清單
├── .env.example           # 環境變數範例
├── docs/                  # 範例文件或上傳文件
├── tests/                 # pytest 測試
├── RunSOP.txt             # 簡短執行 SOP
└── CHANGELOG.md           # 版本紀錄
```

## 環境需求

- Python 3.11+
- Windows/macOS/Linux 皆可
- 第一次執行 RAG 需要下載 embedding model：
  `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- 若使用雲端 LLM，需要對應 provider 的 API key。

## 需要安裝的套件

專案依賴已整理在 `requirements.txt`：

```text
streamlit
openai
anthropic
sentence-transformers
faiss-cpu
pypdf
python-dotenv
numpy
pytest
```

各套件用途：

| 套件 | 用途 |
| --- | --- |
| `streamlit` | Web UI |
| `openai` | OpenAI 與 OpenAI-compatible API 串接 |
| `anthropic` | Claude API 串接 |
| `sentence-transformers` | 產生文件與問題的 embedding |
| `faiss-cpu` | 建立本機向量索引與相似度搜尋 |
| `pypdf` | 讀取 PDF 文字 |
| `python-dotenv` | 讀取 `.env` 環境變數 |
| `numpy` | embedding array 處理 |
| `pytest` | 單元測試 |

## 安裝與執行

### 1. 建立虛擬環境

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. 安裝依賴

```bash
pip install -r requirements.txt
```

第一次安裝或第一次執行會下載 embedding model，檔案可能有數百 MB。

### 3. 設定 API key

複製環境變數範例：

```bash
cp .env.example .env
```

Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

依你要使用的 provider 填入其中一個或多個：

```env
ZAI_API_KEY=your_zai_api_key
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
GROQ_API_KEY=your_groq_api_key
```

若使用 Ollama，本機需先啟動 Ollama server，通常不需要 API key。

### 4. 啟動 Streamlit

```bash
streamlit run app.py
```

瀏覽器開啟：

```text
http://localhost:8501
```

如果 8501 port 被占用：

```bash
streamlit run app.py --server.port 8502
```

## 測試

```bash
pytest tests/ -v
```

注意：

- 測試會用到 `sentence-transformers`，第一次跑需要能連到 Hugging Face 下載 embedding model，或本機已有模型快取。
- 目前測試檔仍包含舊版 `ask_stream()` 測試案例；若新版架構改由 `llm_client.py` 負責 LLM 串流，該測試需要同步更新。

## 重要設定

可在 `config.py` 調整：

| 設定 | 預設值 | 說明 |
| --- | --- | --- |
| `EMBEDDING_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` | 多語 embedding model |
| `CHUNK_SIZE` | `300` | 文件切片大小 |
| `CHUNK_OVERLAP` | `50` | 切片重疊長度 |
| `TOP_K` | `3` | 每次搜尋回傳段落數 |
| `MAX_DISTANCE` | `2.0` | FAISS L2 距離門檻 |
| `MAX_FILE_SIZE_MB` | `10` | 上傳檔案大小上限 |
| `MAX_HISTORY_TURNS` | `5` | 對話歷史保留輪數 |
| `DOCS_DIR` | `docs` | 文件儲存資料夾 |

## side_project:g4 作品集呈現

### 專案名稱

HR Assistant - RAG 文件問答助理

### 一句話介紹

以 Streamlit、FAISS、Sentence Transformers 與 LLM tool use 打造的 HR 文件問答系統，能上傳公司規章並根據文件內容回答問題。

### 專案亮點

- 建立完整 RAG pipeline：文件讀取、切片、embedding、向量索引、語意搜尋。
- 使用 FAISS 做本機向量搜尋，降低查詢延遲與外部資料庫依賴。
- 設計 LLM agent tool-use 流程，讓模型可主動搜尋文件再回答。
- 支援多家 LLM provider，包括 Z.AI、OpenAI、Anthropic、Groq、Ollama。
- 用 Streamlit 快速產品化，提供上傳、聊天、引用來源、匯出紀錄等完整操作流程。

### 技術棧

Python 3.11, Streamlit, Sentence Transformers, FAISS, OpenAI SDK, Anthropic SDK, pypdf, pytest

### 可放履歷的描述

開發一套 HR 文件 RAG 問答助理，支援上傳 TXT/PDF 政策文件，使用 Sentence Transformers 產生多語 embedding，透過 FAISS 建立本機向量索引，並串接多種 LLM provider 產生具來源依據的回答。專案包含 Streamlit UI、agent tool-use 檢索流程、問答紀錄匯出與基礎測試，展示從 AI prototype 到可操作 side project 的完整實作能力。

### 面試可講的技術重點

- 為什麼選 FAISS：適合 side project 與本機 demo，部署簡單，查詢速度快。
- 為什麼做 chunk overlap：避免文件切片切斷語意，提升檢索品質。
- 為什麼使用多 provider client：降低模型供應商綁定，方便切換不同 API。
- RAG 限制：回答品質依賴文件內容、chunk size、embedding model 與 LLM 指令設計。
- 下一步可優化：加入持久化向量庫、重新索引機制、引用段落高亮、權限控管與更完整的 evaluation set。

## 常見問題

### `ModuleNotFoundError`

確認已啟用虛擬環境，並重新安裝：

```bash
pip install -r requirements.txt
```

### 第一次啟動很慢

`sentence-transformers` 需要下載 embedding model。下載完成後會快取，之後啟動會較快。

### 無法連 Hugging Face

請確認網路權限，或先在可連網環境下載模型快取。

### API key 沒有讀到

確認 `.env` 位於專案根目錄，且變數名稱與 provider 對應：

```env
ZAI_API_KEY=...
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GROQ_API_KEY=...
```
