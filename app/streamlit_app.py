"""Streamlit demo for Glance Fashion Search.

Launch:
    pip install streamlit
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import streamlit as st

from glance_search.config import load_config
from glance_search.logging_setup import configure_logging, get_logger
from glance_search.model import ClipModel
from glance_search.pipeline import load_indexes, search

log = get_logger(__name__)

EXAMPLES = [
    "A person in a bright yellow raincoat.",
    "Professional business attire inside a modern office.",
    "Someone wearing a blue shirt sitting on a park bench.",
    "Casual weekend outfit for a city walk.",
    "A red tie and a white shirt in a formal setting.",
]


@st.cache_resource(show_spinner="Loading model + indexes...")
def _load():
    cfg = load_config()
    configure_logging(cfg.log_level)
    loaded = load_indexes(cfg)
    model = ClipModel.get(cfg.model)
    return cfg, loaded, model


def _result_image(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except Exception:
        return b""


def main() -> None:
    st.set_page_config(page_title="Glance Fashion Search", page_icon="*", layout="wide")
    st.title("Glance Fashion Search")
    st.caption("Multimodal fashion + context retrieval  -  text query to top-k images")

    cfg, loaded, model = _load()

    with st.sidebar:
        st.header("Controls")
        query = st.text_input("Query", placeholder="A person in a bright yellow raincoat")
        st.write("Try an example:")
        for ex in EXAMPLES:
            if st.button(ex, key=ex):
                st.session_state["pending_query"] = ex
        top_k = st.slider("Top-K", 1, 20, cfg.retrieval.top_k)
        use_rerank = st.checkbox("Cross-encoder re-ranker", value=cfg.retrieval.use_reranker)
        image_w = st.slider("Image weight", 0.0, 1.0, cfg.retrieval.image_weight, 0.05)
        cap_w = st.slider("Caption weight", 0.0, 1.0, cfg.retrieval.caption_weight, 0.05)
        show_caption = st.checkbox("Show image captions", value=True)
        show_breakdown = st.checkbox("Show score breakdown", value=True)

    pending = st.session_state.pop("pending_query", None)
    query = pending or query
    if not query:
        st.info("Type a query or click an example in the sidebar.")
        return

    new_cfg = Config(
        model=cfg.model,
        index=cfg.index,
        retrieval=replace(
            cfg.retrieval,
            top_k=top_k,
            use_reranker=use_rerank,
            image_weight=image_w,
            caption_weight=cap_w,
        ),
        captions=cfg.captions,
        rerank=cfg.rerank,
        log_level=cfg.log_level,
    )

    with st.spinner("Encoding query and searching..."):
        results = search(query, new_cfg, loaded=loaded, model=model)

    if not results:
        st.warning("No results.")
        return

    st.subheader(f"Top {len(results)} for:  {query}")
    cols = st.columns(min(len(results), 5))
    for col, r in zip(cols * (len(results) // max(len(cols), 1) + 1), results):
        with col:
            data = _result_image(r.path)
            if data:
                st.image(data)
            st.markdown(f"**#{r.rank}**  score=`{r.score:.3f}`")
            if show_breakdown:
                st.caption(f"image: `{r.image_score:.3f}`  caption: `{r.caption_score:.3f}`")
                if r.rerank_score is not None:
                    st.caption(f"rerank: `{r.rerank_score:.3f}`")
            if show_caption and r.caption:
                st.write(f"> {r.caption}")

    with st.expander("Query metadata"):
        st.json({
            "query": query,
            "model": cfg.model.name,
            "pretrained": cfg.model.pretrained,
            "device": cfg.model.device,
            "image_index_ntotal": loaded.image_store.ntotal,
            "caption_index_ntotal": loaded.caption_store.ntotal if loaded.caption_store else 0,
            "captions_in_use": bool(loaded.caption_store),
            "results_returned": len(results),
        })


if __name__ == "__main__":
    main()
