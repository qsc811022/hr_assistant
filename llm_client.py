"""Multi-provider LLM client with agent (tool use) loop."""

import json
import logging
from typing import Callable, Iterator

logger = logging.getLogger(__name__)

# ── Provider presets ──────────────────────────────────────────────────────────

PROVIDER_PRESETS: dict[str, dict] = {
    "Z.AI (GLM)": {
        "type": "openai_compat",
        "base_url": "https://api.z.ai/api/coding/paas/v4/",
        "default_model": "glm-4.7",
        "hint": "Z.AI API key",
    },
    "OpenAI": {
        "type": "openai_compat",
        "base_url": "",
        "default_model": "gpt-4o-mini",
        "hint": "API key 格式：sk-...",
    },
    "Anthropic (Claude)": {
        "type": "anthropic",
        "base_url": "",
        "default_model": "claude-haiku-4-5-20251001",
        "hint": "API key 格式：sk-ant-...",
    },
    "Groq (免費)": {
        "type": "openai_compat",
        "base_url": "https://api.groq.com/openai/v1",
        "default_model": "llama-3.3-70b-versatile",
        "hint": "API key 格式：gsk_...",
    },
    "本機 Ollama": {
        "type": "openai_compat",
        "base_url": "http://localhost:11434/v1",
        "default_model": "llama3.2",
        "hint": "無需 API key，需先啟動 Ollama",
    },
    "自訂（OpenAI 相容）": {
        "type": "openai_compat",
        "base_url": "",
        "default_model": "",
        "hint": "任何相容 OpenAI API 格式的服務",
    },
}

# ── Tool definitions ──────────────────────────────────────────────────────────

_OPENAI_TOOL = {
    "type": "function",
    "function": {
        "name": "search_documents",
        "description": "搜尋已上傳文件中的相關段落，可換不同關鍵字多次呼叫",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜尋關鍵字或問句"}
            },
            "required": ["query"],
        },
    },
}

_ANTHROPIC_TOOL = {
    "name": "search_documents",
    "description": "搜尋已上傳文件中的相關段落，可換不同關鍵字多次呼叫",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "搜尋關鍵字或問句"}
        },
        "required": ["query"],
    },
}

AGENT_SYSTEM_PROMPT = """你是一個智慧文件問答助手，能搜尋使用者上傳的文件來回答問題。

工作流程：
1. 收到問題後，使用 search_documents 工具搜尋相關內容
2. 若結果不夠完整，可換不同關鍵字再搜一次（最多 3 次）
3. 根據搜尋結果整理出清楚的回答

規則：
- 只根據文件內容回答，不捏造資訊
- 找不到相關內容時，誠實告知
- 使用繁體中文，條列清楚"""


# ── LLM Client ────────────────────────────────────────────────────────────────

class LLMClient:
    """Provider-agnostic LLM client with agent (tool use) loop."""

    def __init__(self, preset_name: str, api_key: str, model: str, base_url: str = ""):
        preset = PROVIDER_PRESETS.get(preset_name, PROVIDER_PRESETS["自訂（OpenAI 相容）"])
        self.preset_name = preset_name
        self._ptype: str = preset["type"]
        self._model: str = model or preset["default_model"]
        self._api_key: str = api_key
        self._base_url: str = base_url or preset["base_url"]
        self._sdk: object = None

    def _get_sdk(self):
        if self._sdk is not None:
            return self._sdk
        if self._ptype == "anthropic":
            from anthropic import Anthropic
            self._sdk = Anthropic(api_key=self._api_key)
        else:
            from openai import OpenAI
            kw: dict = {"api_key": self._api_key or "none"}
            if self._base_url:
                kw["base_url"] = self._base_url
            self._sdk = OpenAI(**kw)
        return self._sdk

    def run_agent(
        self,
        question: str,
        search_fn: Callable[[str], list[dict]],
        history: list[dict] | None = None,
    ) -> tuple[Iterator[str], list[dict]]:
        """
        Agent loop: LLM decides what/how to search, then streams the final answer.
        Returns (token_generator, deduplicated_search_results).
        """
        sdk = self._get_sdk()
        clean_history = [h for h in (history or []) if h["role"] in ("user", "assistant")]

        if self._ptype == "anthropic":
            return self._anthropic_loop(sdk, question, search_fn, clean_history)
        else:
            return self._openai_loop(sdk, question, search_fn, clean_history)

    # ── OpenAI / OpenAI-compatible agent loop ─────────────────────────────────

    def _openai_loop(self, sdk, question, search_fn, history):
        messages: list[dict] = [{"role": "system", "content": AGENT_SYSTEM_PROMPT}]
        messages.extend(history[-10:])
        messages.append({"role": "user", "content": question})
        all_results: list[dict] = []

        for _ in range(3):
            resp = sdk.chat.completions.create(
                model=self._model,
                messages=messages,
                tools=[_OPENAI_TOOL],
                tool_choice="auto",
                max_tokens=1024,
            )
            msg = resp.choices[0].message

            if not msg.tool_calls:
                break  # LLM is done searching

            # Execute tool calls and feed results back
            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ],
            })
            for tc in msg.tool_calls:
                query = _extract_tool_query(tc.function.arguments, question)
                logger.info("[Agent] searching: %s", query)
                results = search_fn(query)
                all_results.extend(results)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": _format_context(results),
                })

        # Stream final answer (no tools — prevent re-triggering)
        final = sdk.chat.completions.create(
            model=self._model,
            messages=messages,
            max_tokens=1024,
            stream=True,
        )

        def gen(s=final):
            for chunk in s:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta

        return gen(), _deduplicate(all_results)

    # ── Anthropic agent loop ───────────────────────────────────────────────────

    def _anthropic_loop(self, sdk, question, search_fn, history):
        messages: list[dict] = list(history[-10:])
        messages.append({"role": "user", "content": question})
        all_results: list[dict] = []

        for _ in range(3):
            resp = sdk.messages.create(
                model=self._model,
                max_tokens=1024,
                system=AGENT_SYSTEM_PROMPT,
                tools=[_ANTHROPIC_TOOL],
                messages=messages,
            )

            if resp.stop_reason != "tool_use":
                break  # LLM is done searching

            messages.append({"role": "assistant", "content": resp.content})
            tool_results = []
            for block in resp.content:
                if block.type == "tool_use":
                    query = block.input.get("query", question)
                    logger.info("[Agent] searching: %s", query)
                    results = search_fn(query)
                    all_results.extend(results)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": _format_context(results),
                    })
            messages.append({"role": "user", "content": tool_results})

        # Build clean final prompt (avoids tool-history complications in streaming)
        deduped = _deduplicate(all_results)
        context = _format_context(deduped)
        final_prompt = (
            f"根據以下文件內容回答問題：\n\n{context}\n\n問題：{question}"
            if deduped
            else f"問題：{question}\n\n（文件中未找到相關內容，請如實告知）"
        )
        final_messages = list(history[-6:]) + [{"role": "user", "content": final_prompt}]

        def gen(c=sdk, m=self._model, sys=AGENT_SYSTEM_PROMPT, msgs=final_messages):
            with c.messages.stream(
                model=m,
                max_tokens=1024,
                system=sys,
                messages=msgs,
            ) as s:
                for text in s.text_stream:
                    yield text

        return gen(), deduped


# ── Helpers ───────────────────────────────────────────────────────────────────

def _format_context(results: list[dict]) -> str:
    if not results:
        return "（搜尋未找到相關內容）"
    return "\n\n".join(f"[來源：{r['source']}]\n{r['content']}" for r in results)


def _extract_tool_query(arguments: str, fallback: str) -> str:
    try:
        parsed = json.loads(arguments or "{}")
    except json.JSONDecodeError:
        logger.warning("Invalid tool arguments from model: %s", arguments)
        return fallback

    query = parsed.get("query")
    return query.strip() if isinstance(query, str) and query.strip() else fallback


def _deduplicate(results: list[dict]) -> list[dict]:
    seen: set[tuple] = set()
    out: list[dict] = []
    for r in results:
        key = (r["source"], r["content"][:80])
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out
