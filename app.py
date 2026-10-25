import html
import json
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from backend.btw_handler import handle_btw
from backend.paper_loader import load_arxiv, load_document, load_webpage
from backend.rag_graph import build_graph
from backend.vector_store import add_paper, list_papers

st.set_page_config(page_title="Papeer", page_icon="📚", layout="centered")

# ── Minimal styling (theme colours live in .streamlit/config.toml) ─────────────
st.markdown(
    """
    <style>
    header[data-testid="stHeader"] { background: transparent; }
    #MainMenu, footer, [data-testid="stDecoration"] { display: none; }
    .block-container { padding-top: 3.5rem; max-width: 760px; }

    .thinking { display: flex; flex-direction: column; gap: 0.5rem; padding: 0.35rem 0; max-width: 320px; }
    .thinking span { font-size: 0.85rem; color: #6b7280; }
    .thinking .track { height: 3px; border-radius: 999px; background: #f3f4f6; overflow: hidden; position: relative; }
    .thinking .track::after {
        content: ""; position: absolute; top: 0; left: -40%; width: 40%; height: 100%;
        border-radius: 999px; background: #111111; animation: thinking-slide 1.1s ease-in-out infinite;
    }
    @keyframes thinking-slide { 0% { left: -40%; } 100% { left: 100%; } }

    section[data-testid="stSidebar"] { border-right: 1px solid #e5e7eb; }
    section[data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }
    section[data-testid="stSidebar"] hr { margin: 1rem 0; }

    .sb-label {
        font-size: 0.72rem; font-weight: 600; letter-spacing: 0.06em;
        text-transform: uppercase; color: #6b7280; margin: 0.25rem 0 0.5rem;
    }
    .brand { font-size: 1.05rem; font-weight: 700; letter-spacing: -0.02em; margin-bottom: 1rem; }
    .doc-item {
        font-size: 0.85rem; color: #374151; padding: 0.35rem 0.6rem;
        border: 1px solid #e5e7eb; border-radius: 0.375rem;
        background: #ffffff; margin-bottom: 0.35rem;
        overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    }

    .stButton > button, .stDownloadButton > button {
        font-weight: 500; box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }
    section[data-testid="stSidebar"] .stButton > button {
        justify-content: flex-start; font-size: 0.875rem;
    }
    section[data-testid="stSidebar"] .stButton > button[kind="tertiary"] {
        width: 100%; padding: 0.4rem 0.75rem; color: #374151; border-radius: 0.5rem; box-shadow: none;
    }
    section[data-testid="stSidebar"] .stButton > button[kind="tertiary"]:hover {
        background: #f3f4f6; color: #111111;
    }
    .st-key-chat_list button,
    .st-key-chat_list button div {
        justify-content: flex-start !important; text-align: left !important; min-width: 0;
    }
    .st-key-chat_list button p {
        text-align: left !important; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }
    .st-key-chat_list [data-testid="stVerticalBlock"] { gap: 0.15rem; }

    .hero h1 { font-size: 1.9rem; font-weight: 600; letter-spacing: -0.03em; margin: 0; }
    .hero p { color: #6b7280; font-size: 0.95rem; margin: 0.4rem 0 0; }
    .pills { display: flex; gap: 0.5rem; flex-wrap: wrap; margin: 1.1rem 0 1.75rem; }
    .pill {
        font-size: 0.78rem; color: #374151; padding: 0.25rem 0.65rem;
        border: 1px solid #e5e7eb; border-radius: 999px; background: #ffffff;
    }

    [data-testid="stChatMessage"] { background: transparent; padding: 0.75rem 0; gap: 0.75rem; }
    [data-testid="stChatMessageAvatarUser"],
    [data-testid="stChatMessageAvatarAssistant"] {
        background: #f3f4f6; color: #111111; border: 1px solid #e5e7eb;
    }
    [data-testid="stExpander"] details { border-color: #f0f0f0; }
    [data-testid="stExpander"] summary { font-size: 0.8rem; color: #6b7280; }
    [data-testid="stChatInput"] { box-shadow: 0 1px 3px rgba(0,0,0,0.06); }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_graph():
    return build_graph()


SESSIONS_FILE = Path("sessions.json")
_rename_llm = ChatOpenAI(model="gpt-5-mini")


def load_sessions() -> dict:
    try:
        return json.loads(SESSIONS_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_sessions(sessions_meta: dict) -> None:
    SESSIONS_FILE.write_text(json.dumps(sessions_meta, indent=2), encoding="utf-8")


def _serialize_state(values: dict) -> dict:
    out = {}
    for k, v in values.items():
        if k == "messages":
            out[k] = [
                {
                    "type": type(m).__name__,
                    "content": (
                        m.content[:300]
                        if isinstance(m.content, str)
                        else repr(m.content)[:300]
                    ),
                }
                for m in (v or [])
            ]
        elif k == "retrieved_docs":
            out[k] = [
                {"content": d.page_content[:300], "metadata": d.metadata}
                for d in (v or [])
            ]
        else:
            out[k] = v
    return out


def show_loading(slot, label: str) -> None:
    slot.markdown(
        f'<div class="thinking"><span>{label}</span><div class="track"></div></div>',
        unsafe_allow_html=True,
    )


def next_step_label(node: str, update) -> str | None:
    # Graph "updates" arrive after a node finishes, so describe what runs next.
    update = update if isinstance(update, dict) else {}
    if node == "router":
        return {
            "retrieve": "Searching your sources…",
            "verify_claim": "Checking recent literature…",
        }.get(update.get("route"), "Writing answer…")
    if node in ("agent_node", "query_rewrite"):
        return "Searching your sources…"
    if node == "retrieval":
        return "Reviewing results…"
    if node == "relevancy_check":
        return "Writing answer…" if update.get("is_relevant") else "Refining the search…"
    if node == "verify_claim":
        return "Writing answer…"
    return None


def generate_session_name(first_message: str) -> str:
    try:
        response = _rename_llm.invoke(
            [
                {
                    "role": "system",
                    "content": (
                        "Generate a concise 3-5 word title for a research chat session "
                        "based on the user's first message. Return only the title, "
                        "no punctuation at the end, no quotes."
                    ),
                },
                {"role": "user", "content": first_message[:500]},
            ]
        )
        return response.content.strip()
    except Exception:
        return "New Session"


def maybe_rename_session(session_id: str, first_message: str) -> None:
    if st.session_state.sessions_meta.get(session_id, {}).get("is_named"):
        return
    name = generate_session_name(first_message)
    st.session_state.sessions_meta[session_id]["name"] = name
    st.session_state.sessions_meta[session_id]["is_named"] = True
    save_sessions(st.session_state.sessions_meta)


def create_session() -> str:
    sid = str(uuid.uuid4())
    st.session_state.sessions_meta[sid] = {
        "id": sid,
        "name": "New Session",
        "created_at": datetime.now().isoformat(),
        "is_named": False,
    }
    save_sessions(st.session_state.sessions_meta)
    st.session_state.chats[sid] = []
    st.session_state.turns[sid] = 0
    return sid


def load_session_chats(session_id: str) -> list[dict]:
    config = {"configurable": {"thread_id": session_id}}
    try:
        state = graph.get_state(config)
        if not state or not state.values:
            return []
        chats = []
        turn = 0
        for msg in state.values.get("messages", []):
            type_name = type(msg).__name__
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            if type_name == "HumanMessage":
                chats.append({"role": "user", "content": content})
            elif type_name in ("AIMessage", "AIMessageChunk"):
                turn += 1
                chats.append({"role": "assistant", "content": content, "turn": turn, "graph_state": {}})
        return chats
    except Exception:
        return []


def switch_session(session_id: str) -> None:
    st.session_state.active_session_id = session_id
    if session_id not in st.session_state.chats:
        st.session_state.chats[session_id] = load_session_chats(session_id)
    if session_id not in st.session_state.turns:
        turn_count = sum(1 for m in st.session_state.chats[session_id] if m["role"] == "assistant")
        st.session_state.turns[session_id] = turn_count


graph = get_graph()

# ── Bootstrap ──────────────────────────────────────────────────────────────────
if "sessions_meta" not in st.session_state:
    st.session_state.sessions_meta = load_sessions()
if "chats" not in st.session_state:
    st.session_state.chats = {}
if "turns" not in st.session_state:
    st.session_state.turns = {}
if "active_session_id" not in st.session_state:
    if st.session_state.sessions_meta:
        latest = max(
            st.session_state.sessions_meta.values(),
            key=lambda s: s["created_at"],
        )
        switch_session(latest["id"])
    else:
        sid = create_session()
        st.session_state.active_session_id = sid

active_sid = st.session_state.active_session_id

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="brand">Papeer</div>', unsafe_allow_html=True)
    if st.button("New chat", icon=":material/add:", use_container_width=True, type="primary"):
        new_sid = create_session()
        st.session_state.active_session_id = new_sid
        active_sid = new_sid
        st.rerun()
    st.divider()
    st.markdown('<div class="sb-label">Chats</div>', unsafe_allow_html=True)

    sorted_sessions = sorted(
        st.session_state.sessions_meta.values(),
        key=lambda s: s["created_at"],
        reverse=True,
    )
    with st.container(key="chat_list"):
        for session in sorted_sessions:
            sid = session["id"]
            is_active = sid == st.session_state.active_session_id
            btn_type = "secondary" if is_active else "tertiary"
            if st.button(
                session["name"],
                key=f"sess_{sid}",
                use_container_width=True,
                type=btn_type,
                icon=":material/chat_bubble:" if is_active else ":material/chat_bubble_outline:",
            ):
                if not is_active:
                    switch_session(sid)
                    st.rerun()

    st.divider()
    st.markdown('<div class="sb-label">Add sources</div>', unsafe_allow_html=True)
    tab_files, tab_web, tab_arxiv = st.tabs(["Files", "Web", "ArXiv"])

    # ── Section 1: File upload ─────────────────────────────────────────────────
    with tab_files:
        uploaded_files = st.file_uploader(
            "PDF, TXT, or Markdown",
            type=["pdf", "txt", "md", "markdown"],
            accept_multiple_files=True,
            key=f"uploader_{active_sid}",
            label_visibility="collapsed",
        )
        if st.button("Add Files", use_container_width=True, key="btn_add_files"):
            if uploaded_files:
                processed_key = f"processed_files_{active_sid}"
                if processed_key not in st.session_state:
                    st.session_state[processed_key] = set()
                with st.spinner("Processing files…"):
                    for f in uploaded_files:
                        if f.name in st.session_state[processed_key]:
                            st.info(f"Already loaded: {f.name}")
                            continue
                        suffix = Path(f.name).suffix
                        tmp_path = None
                        try:
                            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                                tmp.write(f.read())
                                tmp_path = tmp.name
                            docs = load_document(tmp_path)
                            for doc in docs:
                                doc.metadata["title"] = Path(f.name).stem
                            add_paper(docs, active_sid)
                            st.session_state[processed_key].add(f.name)
                            st.success(f"Added: {f.name}")
                        except Exception as e:
                            st.error(f"Failed: {f.name} — {e}")
                        finally:
                            if tmp_path:
                                Path(tmp_path).unlink(missing_ok=True)
                st.rerun()
            else:
                st.warning("No files selected.")

    # ── Section 2: Web URL loader ──────────────────────────────────────────────
    with tab_web:
        url_input = st.text_area(
            "URLs (one per line)",
            key=f"url_area_{active_sid}",
            height=80,
            label_visibility="collapsed",
            placeholder="https://example.com/paper",
        )
        if st.button("Load URLs", use_container_width=True, key="btn_load_urls"):
            urls = [u.strip() for u in url_input.splitlines() if u.strip()]
            if urls:
                with st.spinner("Loading web pages…"):
                    for url in urls:
                        try:
                            docs = load_webpage(url)
                            add_paper(docs, active_sid)
                            st.success(f"Loaded: {url[:60]}")
                        except Exception as e:
                            st.error(f"Failed: {url[:60]} — {e}")
                st.rerun()
            else:
                st.warning("Enter at least one URL.")

    # ── Section 3: ArXiv loader ────────────────────────────────────────────────
    with tab_arxiv:
        arxiv_title = st.text_input(
            "Paper title or ArXiv ID",
            key=f"arxiv_input_{active_sid}",
            label_visibility="collapsed",
            placeholder="1706.03762  or  Attention Is All You Need",
        )
        if st.button("Load ArXiv Paper", use_container_width=True, key="btn_load_arxiv"):
            if arxiv_title.strip():
                with st.spinner("Loading from ArXiv…"):
                    try:
                        docs = load_arxiv(arxiv_title.strip())
                        add_paper(docs, active_sid)
                        loaded_title = docs[0].metadata.get("title") if docs else arxiv_title.strip()
                        st.success(f"Loaded: {loaded_title}")
                    except Exception as e:
                        st.error(f"Failed: {e}")
                st.rerun()
            else:
                st.warning("Enter a paper title or ArXiv ID.")

    # ── Loaded Documents list ──────────────────────────────────────────────────
    st.divider()
    st.markdown('<div class="sb-label">In this chat</div>', unsafe_allow_html=True)
    try:
        doc_titles = list_papers(active_sid)
    except Exception:
        doc_titles = None
    if doc_titles is None:
        st.caption("Could not load document list — try refreshing.")
    elif doc_titles:
        for title in doc_titles:
            st.markdown(f'<div class="doc-item" title="{html.escape(str(title))}">{html.escape(str(title))}</div>', unsafe_allow_html=True)
    else:
        st.caption("No documents yet.")

# ── Page header ────────────────────────────────────────────────────────────────
if not st.session_state.chats.get(active_sid):
    st.markdown(
        """
        <div class="hero">
          <h1>What are you researching?</h1>
          <p>Ask questions about your papers, check claims against recent work, or search the web.</p>
        </div>
        <div class="pills">
          <span class="pill">Ask your papers</span>
          <span class="pill">Verify a claim</span>
          <span class="pill">Search the web</span>
          <span class="pill">/btw for a quick side question</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f'<div class="hero"><h1 style="font-size:1.35rem">'
        f'{html.escape(st.session_state.sessions_meta.get(active_sid, {}).get("name", "Chat"))}'
        f"</h1></div>",
        unsafe_allow_html=True,
    )
    st.write("")

# ── Chat display ───────────────────────────────────────────────────────────────
for msg in st.session_state.chats.get(active_sid, []):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            with st.expander(f"Details · turn {msg['turn']}", expanded=False):
                st.json(msg["graph_state"])

# ── Chat input ─────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask anything…"):
    is_btw = prompt.strip().lower().startswith("/btw")

    if is_btw:
        query = prompt.strip()[4:].strip()

        with st.chat_message("user"):
            st.markdown(prompt)
            st.caption("Side channel — not saved to session history.")

        with st.chat_message("assistant"):
            if not query:
                st.markdown("Please add a question after `/btw`, e.g. `/btw What is attention?`")
            else:
                placeholder = st.empty()
                show_loading(placeholder, "Thinking…")
                response_text = ""
                for chunk in handle_btw(query):
                    response_text += chunk
                    placeholder.markdown(response_text + "▌")
                placeholder.markdown(response_text)
            st.caption("Side channel — not saved to session history.")

    else:
        if active_sid not in st.session_state.chats:
            st.session_state.chats[active_sid] = []
        if active_sid not in st.session_state.turns:
            st.session_state.turns[active_sid] = 0

        is_first_message = len(st.session_state.chats[active_sid]) == 0

        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.chats[active_sid].append({"role": "user", "content": prompt})
        st.session_state.turns[active_sid] += 1
        current_turn = st.session_state.turns[active_sid]

        if is_first_message:
            maybe_rename_session(active_sid, prompt)

        input_state = {
            "messages": [HumanMessage(content=prompt)],
            "session_id": active_sid,
            "query": prompt,
            "route": None,
            "retrieved_docs": [],
            "retrieval_attempts": 0,
            "claim_verdict": None,
            "claim_source": None,
            "superseding_papers": [],
            "answer": None,
            "is_relevant": None,
            "rewrite_count": 0,
        }
        config = {"configurable": {"thread_id": active_sid}}

        with st.chat_message("assistant"):
            placeholder = st.empty()
            show_loading(placeholder, "Thinking…")
            response_text = ""

            for mode, payload in graph.stream(input_state, config, stream_mode=["messages", "updates"]):
                if mode == "updates":
                    if not response_text:
                        for node, update in (payload or {}).items():
                            label = next_step_label(node, update)
                            if label:
                                show_loading(placeholder, label)
                    continue
                chunk, metadata = payload
                if (
                    metadata.get("langgraph_node") == "generate_answer"
                    and hasattr(chunk, "content")
                    and chunk.content
                ):
                    response_text += chunk.content
                    placeholder.markdown(response_text + "▌")

            if not response_text:
                final_values = graph.get_state(config).values
                response_text = final_values.get("answer") or "No response generated."

            placeholder.markdown(response_text)

            final_values = graph.get_state(config).values
            state_snapshot = _serialize_state(final_values)

            with st.expander(f"Details · turn {current_turn}", expanded=False):
                st.json(state_snapshot)

        st.session_state.chats[active_sid].append(
            {
                "role": "assistant",
                "content": response_text,
                "graph_state": state_snapshot,
                "turn": current_turn,
            }
        )

        if is_first_message:
            st.rerun()