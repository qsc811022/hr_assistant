import streamlit as st
from rag_core import HRKnowledgeBase
from pathlib import Path

# ── 頁面設定 ──────────────────────────
st.set_page_config(
    page_title="HR 小幫手",
    page_icon="🤝",
    layout="wide",
)

# ── 初始化知識庫（只建立一次）─────────
if "kb" not in st.session_state:
    st.session_state.kb = HRKnowledgeBase()
    st.session_state.loaded_files = []
    st.session_state.messages = []

    # 自動載入 docs/ 資料夾裡的所有文件
    docs_dir = Path("docs")
    if docs_dir.exists():
        for f in docs_dir.glob("*.txt"):
            count = st.session_state.kb.add_document(str(f))
            st.session_state.loaded_files.append(f.name)

# ── 左側欄：文件管理 ───────────────────
with st.sidebar:
    st.title("📂 文件管理")

    # 顯示已載入文件
    if st.session_state.loaded_files:
        st.success(f"已載入 {len(st.session_state.loaded_files)} 份文件")
        for f in st.session_state.loaded_files:
            st.caption(f"✓ {f}")
    else:
        st.warning("尚未載入任何文件")

    st.divider()

    # 上傳新文件
    uploaded = st.file_uploader(
        "上傳公司文件",
        type=["txt", "pdf"],
        accept_multiple_files=True,
    )
    if uploaded:
        for file in uploaded:
            if file.name not in st.session_state.loaded_files:
                # 存到 docs/ 資料夾
                save_path = Path("docs") / file.name
                save_path.parent.mkdir(exist_ok=True)
                save_path.write_bytes(file.read())

                # 載入到知識庫
                count = st.session_state.kb.add_document(str(save_path))
                st.session_state.loaded_files.append(file.name)
                st.success(f"已載入：{file.name}（{count} 個段落）")
                st.rerun()

    st.divider()

    # 清除對話
    if st.button("🗑️ 清除對話記錄"):
        st.session_state.messages = []
        st.rerun()

    st.caption("HR 小幫手 v1.0")

# ── 主區域：聊天介面 ───────────────────
st.title("🤝 HR 小幫手")
st.caption("有任何關於請假、薪資、福利的問題，直接問我！")

# 顯示對話歷史
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            st.caption(f"參考來源：{', '.join(msg['sources'])}")

# 輸入框
question = st.chat_input("請問有什麼我可以幫你的？例如：年假有幾天？")

if question:
    # 顯示使用者問題
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # AI 串流回答
    with st.chat_message("assistant"):
        if not st.session_state.loaded_files:
            st.warning("請先在左側上傳公司文件！")
        else:
            generator, sources = st.session_state.kb.ask_stream(question)

            # 串流顯示
            response = st.write_stream(generator)

            # 顯示來源
            if sources:
                st.caption(f"參考來源：{', '.join(sources)}")

            # 存入歷史
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "sources": sources,
            })