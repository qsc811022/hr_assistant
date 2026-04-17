"""HR Assistant Streamlit application."""

import logging
from datetime import datetime
from pathlib import Path

import streamlit as st

from config import APP_TITLE, APP_VERSION, DOCS_DIR, FAQ_QUESTIONS
from rag_core import DocumentLoadError, HRKnowledgeBase

logging.basicConfig(level=logging.INFO)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🤝",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .doc-card {
        background: #f0f7f0;
        border-left: 3px solid #4CAF50;
        padding: 8px 12px;
        margin: 4px 0;
        border-radius: 0 6px 6px 0;
        font-size: 14px;
    }
    .welcome-box {
        text-align: center;
        padding: 60px 20px;
        color: #888;
    }
    .low-relevance {
        color: #e67e22;
        font-size: 12px;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ── Session init ──────────────────────────────────────────────────────────────
def _init_session() -> None:
    if "kb" in st.session_state:
        return

    st.session_state.kb = HRKnowledgeBase()
    st.session_state.loaded_files = {}
    st.session_state.messages = []
    st.session_state.pending_question = None

    docs_dir = Path(DOCS_DIR)
    if docs_dir.exists():
        for f in sorted(docs_dir.glob("*.txt")):
            try:
                count = st.session_state.kb.add_document(str(f))
                st.session_state.loaded_files[f.name] = {
                    "chunk_count": count,
                    "loaded_at": datetime.now().strftime("%H:%M"),
                }
            except DocumentLoadError as exc:
                st.warning(f"⚠️ 無法載入 {f.name}：{exc}")


# ── Sidebar ───────────────────────────────────────────────────────────────────
def _render_sidebar() -> None:
    with st.sidebar:
        st.title("📂 文件管理")

        if st.session_state.loaded_files:
            st.success(f"✅ 已載入 {len(st.session_state.loaded_files)} 份文件")
            for fname, meta in st.session_state.loaded_files.items():
                st.markdown(
                    f'<div class="doc-card"><b>{fname}</b><br>'
                    f'<small>📦 {meta["chunk_count"]} 段落　⏱ {meta["loaded_at"]}</small></div>',
                    unsafe_allow_html=True,
                )
        else:
            st.warning("⚠️ 尚未載入任何文件")

        st.divider()

        uploaded = st.file_uploader(
            "上傳公司文件",
            type=["txt", "pdf"],
            accept_multiple_files=True,
            help=f"支援 .txt 與 .pdf，單檔最大 10 MB",
        )
        if uploaded:
            for file in uploaded:
                if file.name not in st.session_state.loaded_files:
                    save_path = Path(DOCS_DIR) / file.name
                    save_path.parent.mkdir(exist_ok=True)
                    save_path.write_bytes(file.read())
                    try:
                        with st.spinner(f"正在解析 {file.name}…"):
                            count = st.session_state.kb.add_document(str(save_path))
                        st.session_state.loaded_files[file.name] = {
                            "chunk_count": count,
                            "loaded_at": datetime.now().strftime("%H:%M"),
                        }
                        st.success(f"✅ {file.name}（{count} 段落）")
                        st.rerun()
                    except DocumentLoadError as exc:
                        st.error(f"❌ {exc}")

        st.divider()

        st.subheader("💡 常見問題")
        for q in FAQ_QUESTIONS:
            if st.button(q, use_container_width=True, key=f"faq_{q}"):
                st.session_state.pending_question = q
                st.rerun()

        st.divider()

        col_clear, col_export = st.columns(2)
        with col_clear:
            if st.button("🗑️ 清除對話", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
        with col_export:
            if st.session_state.messages:
                st.download_button(
                    "💾 匯出",
                    data=_export_markdown(),
                    file_name=f"hr_chat_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                    mime="text/markdown",
                    use_container_width=True,
                )

        st.caption(f"{APP_TITLE} v{APP_VERSION}")


# ── Helpers ───────────────────────────────────────────────────────────────────
def _export_markdown() -> str:
    lines = [
        f"# HR 小幫手對話記錄\n",
        f"匯出時間：{datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n",
    ]
    for msg in st.session_state.messages:
        role_label = "**員工**" if msg["role"] == "user" else "**HR 小幫手**"
        lines.append(f"{role_label}：{msg['content']}\n")
        if msg.get("sources"):
            unique = list({r["source"] for r in msg["sources"]})
            lines.append(f"> 參考來源：{', '.join(unique)}\n")
        lines.append("\n")
    return "".join(lines)


def _render_message(msg: dict) -> None:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            unique = list({r["source"] for r in msg["sources"]})
            st.caption("📄 參考來源：" + "　·　".join(unique))


def _handle_question(question: str) -> None:
    question = question.strip()
    if not question:
        return

    question = question[:500]  # enforce max length

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if not st.session_state.loaded_files:
            st.warning("⚠️ 請先在左側上傳公司文件！")
            return

        try:
            with st.spinner("搜尋相關文件中…"):
                history = [
                    m for m in st.session_state.messages[:-1]
                    if m["role"] in ("user", "assistant")
                ]
                generator, results = st.session_state.kb.ask_stream(
                    question, history=history
                )

            if results:
                avg_relevance = sum(r["relevance"] for r in results) / len(results)
                if avg_relevance < 0.3:
                    st.markdown(
                        '<p class="low-relevance">⚠️ 相關度較低，建議直接聯絡 HR 確認</p>',
                        unsafe_allow_html=True,
                    )

            response = st.write_stream(generator)

            if results:
                unique = list({r["source"] for r in results})
                st.caption("📄 參考來源：" + "　·　".join(unique))

            st.session_state.messages.append(
                {"role": "assistant", "content": response, "sources": results}
            )

        except Exception as exc:
            error_msg = f"抱歉，查詢時發生錯誤：{exc}"
            st.error(error_msg)
            st.session_state.messages.append(
                {"role": "assistant", "content": error_msg, "sources": []}
            )


# ── Main ──────────────────────────────────────────────────────────────────────
_init_session()
_render_sidebar()

st.title(f"🤝 {APP_TITLE}")
st.caption("有任何關於請假、薪資、福利的問題，直接問我！")

if not st.session_state.messages:
    st.markdown(
        """
        <div class="welcome-box">
            <div style="font-size:64px;margin-bottom:12px">🤝</div>
            <h3 style="color:#555">你好！我是 HR 小幫手</h3>
            <p>我能幫你查詢公司規定、假別制度、薪資福利等資訊</p>
            <p style="font-size:13px">在下方輸入問題，或點選左側常見問題快速提問</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    for msg in st.session_state.messages:
        _render_message(msg)

# FAQ shortcut triggered from sidebar button
if st.session_state.get("pending_question"):
    pending = st.session_state.pending_question
    st.session_state.pending_question = None
    _handle_question(pending)

# Direct chat input
question = st.chat_input("請問有什麼我可以幫你的？例如：年假有幾天？")
if question:
    _handle_question(question)
