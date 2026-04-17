# 🤝 HR 小幫手

> 基於 RAG（檢索增強生成）技術的企業 HR 問答助手——讓員工隨時自助查詢公司規定

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📖 簡介

HR 小幫手是一個本地部署的 AI 問答系統，員工可以用自然語言詢問任何關於公司政策的問題——請假規定、薪資福利、績效考核等，系統會從公司文件中精準找到答案，並透過 GLM-4.7 大型語言模型以串流方式生成清晰回覆。

**適用情境：** 公司內部員工自助服務、HR 部門文件知識庫、企業政策 QA 系統

---

## ✨ 功能特色

| 功能 | 說明 |
|------|------|
| 📄 **多文件管理** | 支援 `.txt` / `.pdf` 上傳，自動解析並建立向量索引 |
| 🔍 **語義搜尋** | 多語言 Sentence Transformer 模型，精準匹配相關段落 |
| 💬 **串流回覆** | Token-by-token 即時輸出，附來源引用 |
| 💡 **常見問題快捷** | 一鍵提問，覆蓋最常見 HR 查詢場景 |
| 📊 **相關度提示** | 搜尋結果信心分數，低相關時主動提醒 |
| 💾 **對話匯出** | 一鍵下載 Markdown 格式的完整對話記錄 |
| 🔄 **重試機制** | API 限流自動指數退避重試，連線穩定 |
| 🧪 **單元測試** | 核心 RAG 引擎完整測試覆蓋 |

---

## 🏗️ 系統架構

```
使用者問題
    │
    ▼
┌─────────────────────────────────────┐
│          Streamlit UI (app.py)       │
│  側欄：文件管理 / FAQ / 匯出         │
│  主區：聊天介面 / 串流回覆           │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│       HRKnowledgeBase (rag_core.py)  │
│                                      │
│  add_document()                      │
│    └─ 讀取 TXT/PDF                  │
│    └─ 文字分塊（300字/50重疊）       │
│    └─ Sentence Transformer 向量化    │
│    └─ 建立 FAISS 索引               │
│                                      │
│  search()                            │
│    └─ 問題向量化                     │
│    └─ FAISS L2 相似度搜尋           │
│    └─ 相關度分數過濾                 │
│                                      │
│  ask_stream()                        │
│    └─ 組合 Context + 對話歷史        │
│    └─ 呼叫 GLM-4.7 API（串流）      │
│    └─ 自動重試（指數退避）           │
└─────────────────────────────────────┘
               │
               ▼
         Z.AI / GLM-4.7
```

---

## 🚀 快速開始

### 1. 環境需求

- Python 3.11+
- [Z.AI API 金鑰](https://api.z.ai)

### 2. 安裝依賴

```bash
git clone <your-repo-url>
cd hr_assistant
pip install -r requirements.txt
```

### 3. 設定環境變數

複製 `.env.example` 並填入金鑰：

```bash
cp .env.example .env
```

`.env` 內容：

```env
ZAI_API_KEY=your_api_key_here
```

### 4. 啟動應用

```bash
streamlit run app.py
```

瀏覽器開啟 `http://localhost:8501` 即可使用。

---

## 📁 專案結構

```
hr_assistant/
├── app.py              # Streamlit UI 主程式
├── rag_core.py         # RAG 引擎（文件載入、搜尋、LLM 呼叫）
├── config.py           # 集中管理所有參數與常數
├── requirements.txt    # Python 依賴
├── .env                # API 金鑰（不提交 git）
├── docs/
│   ├── company_policy.txt   # 範例：請假與薪資福利政策
│   └── work_rules.txt       # 範例：工作規範與績效考核
└── tests/
    └── test_rag_core.py     # HRKnowledgeBase 單元測試
```

---

## ⚙️ 參數設定

所有可調整的參數集中於 [config.py](config.py)：

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `MODEL_NAME` | `glm-4.7` | LLM 模型名稱 |
| `CHUNK_SIZE` | `300` | 文件分塊大小（字元數） |
| `CHUNK_OVERLAP` | `50` | 相鄰分塊重疊長度 |
| `TOP_K` | `3` | 每次搜尋取回的最相關段落數 |
| `MAX_DISTANCE` | `2.0` | FAISS L2 距離過濾閾值 |
| `MAX_HISTORY_TURNS` | `5` | 帶入 LLM 的對話歷史輪數 |
| `MAX_FILE_SIZE_MB` | `10` | 單檔最大上傳大小 |
| `FAQ_QUESTIONS` | 6 個預設問題 | 側欄常見問題快捷清單 |

---

## 🧪 執行測試

```bash
pytest tests/ -v
```

測試覆蓋範圍：
- 文件載入（正常、檔案不存在、空白檔案、多文件累積）
- 搜尋（空知識庫、結果結構、相關度範圍、top_k 限制）
- 串流回覆（無文件時的 fallback 行為）

---

## 🛠️ 技術棧

- **[Streamlit](https://streamlit.io)** — 前端介面框架
- **[Sentence Transformers](https://www.sbert.net)** — 多語言語義向量化（`paraphrase-multilingual-MiniLM-L12-v2`）
- **[FAISS](https://github.com/facebookresearch/faiss)** — 高效向量相似度搜尋
- **[GLM-4.7 via Z.AI](https://api.z.ai)** — 大型語言模型（OpenAI 相容 API）
- **[pypdf](https://pypdf.readthedocs.io)** — PDF 文字擷取

---

## 📄 License

MIT
