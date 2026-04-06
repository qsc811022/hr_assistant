import os
import numpy as np
import faiss
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

client = OpenAI(
    api_key=os.getenv("ZAI_API_KEY"),
    base_url="https://api.z.ai/api/coding/paas/v4/"
)

embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

SYSTEM_PROMPT = """你是公司的 HR 小幫手，專門回答員工關於公司規定的問題。

規則：
- 只根據提供的文件內容回答
- 如果文件沒有提到，說「這部分文件沒有記載，建議直接聯絡 HR 部門」
- 回答要簡潔清楚，條列式呈現
- 用繁體中文回答
- 回答後說明參考的文件段落"""

class HRKnowledgeBase:
    def __init__(self):
        self.chunks = []
        self.sources = []     # 記錄每個 chunk 來自哪個檔案
        self.index = None

    def add_document(self, path: str) -> int:
        """載入一份文件"""
        path = Path(path)
        if not path.exists():
            return 0

        # 讀取文件
        if path.suffix == ".pdf":
            from pypdf import PdfReader
            reader = PdfReader(str(path))
            text = "\n".join(p.extract_text() for p in reader.pages)
        else:
            text = path.read_text(encoding="utf-8")

        # 切塊
        chunk_size, overlap = 300, 50
        new_chunks = []
        start = 0
        while start < len(text):
            chunk = text[start:start + chunk_size]
            new_chunks.append(chunk)
            self.sources.append(path.name)   # 記錄來源檔名
            start += chunk_size - overlap

        self.chunks.extend(new_chunks)

        # 重建向量索引
        self._rebuild_index()
        return len(new_chunks)

    def _rebuild_index(self):
        """重新建立 FAISS 索引"""
        if not self.chunks:
            return
        embeddings = embedder.encode(self.chunks)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.array(embeddings, dtype="float32"))

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """搜尋最相關段落，回傳內容和來源"""
        if self.index is None:
            return []
        query_vec = embedder.encode([query])
        distances, indices = self.index.search(
            np.array(query_vec, dtype="float32"), top_k
        )
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.chunks):
                results.append({
                    "content": self.chunks[idx],
                    "source": self.sources[idx],
                    "distance": float(dist),
                })
        return results

    def ask_stream(self, question: str):
        """查詢並串流回答，回傳 (generator, sources)"""
        # 找相關段落
        results = self.search(question, top_k=3)
        if not results:
            def empty():
                yield "目前沒有載入任何文件，請先上傳公司規定文件。"
            return empty(), []

        # 整理參考內容
        context = "\n\n".join(
            f"【來源：{r['source']}】\n{r['content']}"
            for r in results
        )
        sources = list({r["source"] for r in results})

        # 呼叫 AI 串流回答
        stream = client.chat.completions.create(
            model="glm-4.7",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"參考資料：\n{context}\n\n問題：{question}"},
            ],
            max_tokens=1024,
            stream=True,
        )

        def generator():
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta

        return generator(), sources