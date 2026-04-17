"""Document Q&A Assistant — upload documents, configure any LLM, ask questions."""

import logging
import os
from datetime import datetime
from pathlib import Path

import streamlit as st

from config import APP_TITLE, APP_VERSION, DOCS_DIR
from llm_client import PROVIDER_PRESETS, LLMClient
from rag_core import DocumentLoadError, HRKnowledgeBase

logging.basicConfig(level=logging.INFO)

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .doc-card {
        background: #f0f7f0;
        border-left: 3px solid #4CAF50;
        padding: 8px 12px;
        margin: 4px 0;
        border-radius: 0 6px 6px 0;
        font-size: 14px;
    }
    .provider-badge {
        display: inline-block;
        background: #e8f4fd;
        color: #1a6fa0;
        border-radius: 10px;
        padding: 2px 10px;
        font-size: 12px;
    }
</style>
""", unsafe_allow_html=True)


# ── Session init ──────────────────────────────────────────────────────────────
def _init_session() -> None:
    if "kb" not in st.session_state:
        st.session_state.kb = HRKnowledgeBase()
        st.session_state.loaded_files = {}
        st.session_state.messages = []
    if "llm_client" not in st.session_state:
        st.session_state.llm_client = None
    if "provider_preset" not in st.session_state:
        st.session_state.provider_preset = list(PROVIDER_PRESETS.keys())[0]


# ── Sidebar ───────────────────────────────────────────────────────────────────
def _render_sidebar() -> None:
    with st.sidebar:
        # ── AI 設定 ──────────────────────────────────────────────────────────
        st.subheader("⚙️ AI 設定")

        preset_name = st.selectbox(
            "服務商",
            list(PROVIDER_PRESETS.keys()),
            index=list(PROVIDER_PRESETS.keys()).index(st.session_state.provider_preset),
            label_visibility="collapsed",
        )
        preset = PROVIDER_PRESETS[preset_name]
        st.caption(f"💡 {preset['hint']}")

        api_key = st.text_input(
            "API Key",
            type="password",
            placeholder="貼上你的 API key",
            value=_get_env_key(preset_name),
        )

        model = st.text_input(
            "模型名稱",
            value=preset["default_model"],
            placeholder=preset["default_model"],
        )

        base_url = ""
        if preset["type"] == "openai_compat":
            base_url = st.text_input(
                "Base URL",
                value=preset["base_url"],
                placeholder="https://api.example.com/v1",
            )

        if st.button("✅ 套用設定", use_container_width=True, type="primary"):
            if not api_key and preset_name != "本機 Ollama":
                st.error("請輸入 API Key")
            else:
                st.session_state.llm_client = LLMClient(preset_name, api_key, model, base_url)
                st.session_state.provider_preset = preset_name
                st.success(f"已套用：{preset_name} / {model}")

        # 顯示目前啟用的服務商
        if st.session_state.llm_client:
            lc = st.session_state.llm_client
            st.markdown(
                f'<span class="provider-badge">🟢 {lc.preset_name} · {lc._model}</span>',
                unsafe_allow_html=True,
            )

        st.divider()

        # ── 已上傳文件 ────────────────────────────────────────────────────────
        st.subheader("📂 已上傳文件")

        if st.session_state.loaded_files:
            for fname, meta in st.session_state.loaded_files.items():
                st.markdown(
                    f'<div class="doc-card"><b>{fname}</b><br>'
                    f'<small>📦 {meta["chunk_count"]} 段落　⏱ {meta["loaded_at"]}</small></div>',
                    unsafe_allow_html=True,
                )
        else:
            st.caption("尚未上傳任何文件")

        st.divider()

        col_clear, col_export = st.columns(2)
        with col_clear:
            if st.button("🗑️ 清除對話", use_container_width=True,
                         disabled=not st.session_state.messages):
                st.session_state.messages = []
                st.rerun()
        with col_export:
            if st.session_state.messages:
                st.download_button(
                    "💾 匯出",
                    data=_export_markdown(),
                    file_name=f"qa_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                    mime="text/markdown",
                    use_container_width=True,
                )

        st.caption(f"{APP_TITLE} v{APP_VERSION}")


# ── Upload page（無文件時）────────────────────────────────────────────────────
def _render_upload_page() -> None:
    st.title(f"📄 {APP_TITLE}")

    st.markdown("""
    <div style="text-align:center;padding:50px 20px;color:#888">
        <div style="font-size:56px;margin-bottom:16px">📂</div>
        <h3 style="color:#555">請先上傳你的文件</h3>
        <p>支援 <b>.txt</b> 與 <b>.pdf</b>，上傳後即可提問</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "選擇或拖曳文件",
        type=["txt", "pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if uploaded:
        _process_uploads(uploaded)


# ── Chat page（有文件後）──────────────────────────────────────────────────────
def _render_chat_page() -> None:
    st.title(f"📄 {APP_TITLE}")

    # 補充上傳（側欄底部）
    with st.sidebar:
        st.subheader("➕ 新增文件")
        extra = st.file_uploader(
            "上傳更多文件",
            type=["txt", "pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            key="extra_upload",
        )
        if extra:
            _process_uploads(extra)

    # 對話歷史
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                unique = list({r["source"] for r in msg["sources"]})
                st.caption("📄 參考來源：" + "　·　".join(unique))

    # 輸入框
    question = st.chat_input("輸入問題，AI 將搜尋你的文件後回答…")
    if question:
        _handle_question(question)


# ── 處理上傳 ──────────────────────────────────────────────────────────────────
def _process_uploads(files) -> None:
    docs_dir = Path(DOCS_DIR)
    docs_dir.mkdir(exist_ok=True)
    any_new = False

    for file in files:
        if file.name in st.session_state.loaded_files:
            continue
        save_path = docs_dir / file.name
        save_path.write_bytes(file.read())
        try:
            with st.spinner(f"正在解析 {file.name}…"):
                count = st.session_state.kb.add_document(str(save_path))
            st.session_state.loaded_files[file.name] = {
                "chunk_count": count,
                "loaded_at": datetime.now().strftime("%H:%M"),
            }
            st.toast(f"✅ {file.name} 載入完成（{count} 段落）")
            any_new = True
        except DocumentLoadError as exc:
            st.error(f"❌ {file.name}：{exc}")

    if any_new:
        st.rerun()


# ── 問答處理 ──────────────────────────────────────────────────────────────────
def _handle_question(question: str) -> None:
    question = question.strip()[:500]
    if not question:
        return

    if not st.session_state.llm_client:
        st.warning("⚠️ 請先在左側「AI 設定」填入 API Key 並點擊「套用設定」")
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            history = [m for m in st.session_state.messages[:-1]
                       if m["role"] in ("user", "assistant")]

            with st.spinner("Agent 搜尋文件中…"):
                generator, results = st.session_state.llm_client.run_agent(
                    question,
                    st.session_state.kb.search,
                    history=history,
                )

            response = st.write_stream(generator)

            if results:
                unique = list({r["source"] for r in results})
                st.caption("📄 參考來源：" + "　·　".join(unique))

            st.session_state.messages.append(
                {"role": "assistant", "content": response, "sources": results}
            )

        except Exception as exc:
            err = f"發生錯誤：{exc}"
            st.error(err)
            st.session_state.messages.append(
                {"role": "assistant", "content": err, "sources": []}
            )


# ── 匯出 ──────────────────────────────────────────────────────────────────────
def _export_markdown() -> str:
    lines = [f"# 問答記錄\n\n匯出時間：{datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"]
    for msg in st.session_state.messages:
        label = "**你**" if msg["role"] == "user" else "**AI**"
        lines.append(f"{label}：{msg['content']}\n")
        if msg.get("sources"):
            unique = list({r["source"] for r in msg["sources"]})
            lines.append(f"> 來源：{', '.join(unique)}\n")
        lines.append("\n")
    return "".join(lines)


def _get_env_key(preset_name: str) -> str:
    """Pre-fill API key from environment variables if available."""
    mapping = {
        "Z.AI (GLM)": "ZAI_API_KEY",
        "OpenAI": "OPENAI_API_KEY",
        "Anthropic (Claude)": "ANTHROPIC_API_KEY",
        "Groq (免費)": "GROQ_API_KEY",
    }
    env_var = mapping.get(preset_name, "")
    return os.getenv(env_var, "") if env_var else ""


# ── Main ──────────────────────────────────────────────────────────────────────
_init_session()
_render_sidebar()

if not st.session_state.loaded_files:
    _render_upload_page()
else:
    _render_chat_page()
