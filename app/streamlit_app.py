"""Streamlit demo for Glance Fashion Search - top-class edition.

Features:
- Text query with example buttons
- Image-to-image search (upload an image, find visually similar products)
- Side-by-side controls for top-k, image weight, caption weight, rerank
- Score breakdown per result
- Query attribute breakdown (which axes the query hits)
- Aggregation metrics panel

Launch:
    pip install streamlit
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import streamlit as st
from PIL import Image

from glance_search.attributes import parse_query, query_axis_tags
from glance_search.config import Config, load_config
from glance_search.embedder import open_image
from glance_search.logging_setup import configure_logging, get_logger
from glance_search.metrics import aggregate_metrics
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


def _image_to_image_search(
    uploaded_file,
    cfg: Config,
    loaded,
    model,
    top_k: int,
    image_weight: float,
    caption_weight: float,
    use_rerank: bool,
    use_captions: bool,
    expand_q: bool,
) -> list:
    """Encode the uploaded image with the CLIP image encoder and search."""
    img = Image.open(uploaded_file).convert("RGB")
    with st.spinner("Encoding uploaded image..."):
        feats = model.encode_images([img]).cpu().numpy().astype("float32")
    new_cfg = Config(
        model=cfg.model,
        index=cfg.index,
        retrieval=replace(
            cfg.retrieval,
            top_k=top_k,
            image_weight=image_weight,
            caption_weight=caption_weight,
            use_reranker=use_rerank,
            use_captions=use_captions,
            expand_queries=expand_q,
        ),
        captions=cfg.captions,
        rerank=cfg.rerank,
        log_level=cfg.log_level,
    )
    scores, indices = loaded.image_store.search(feats, top_k * 4)
    candidates = {}
    for r, (s, i) in enumerate(zip(scores[0], indices[0])):
        i = int(i)
        if i < 0 or i >= len(loaded.image_paths):
            continue
        candidates.setdefault(i, {"image": 0.0, "caption": 0.0})
        if float(s) > candidates[i]["image"]:
            candidates[i]["image"] = float(s)
        if r < top_k * 4:
            candidates[i]["image_rank"] = min(candidates[i].get("image_rank", 10**9), r + 1)

    from glance_search.pipeline import SearchResult, _rrf_fuse

    image_ranks = {i: v["image_rank"] for i, v in candidates.items() if "image_rank" in v}
    hybrid = _rrf_fuse(image_ranks, {}, image_w=image_weight, caption_w=0.0)
    sorted_idx = sorted(hybrid.keys(), key=lambda x: hybrid[x], reverse=True)[:top_k]
    results = []
    for rank, i in enumerate(sorted_idx, start=1):
        path = loaded.image_paths[i]
        results.append(
            SearchResult(
                path=path,
                score=float(hybrid[i]),
                rank=rank,
                image_score=float(candidates[i]["image"]),
                caption_score=0.0,
                image_rank=image_ranks.get(i),
                caption=loaded.captions.get(str(path)),
            )
        )
    return results


def main() -> None:
    st.set_page_config(page_title="Glance Fashion Search", page_icon="*", layout="wide")
    st.title("Glance Fashion Search")
    st.caption("Multimodal fashion + context retrieval — text query or image upload")

    cfg, loaded, model = _load()

    with st.sidebar:
        st.header("Controls")
        mode = st.radio("Mode", ["Text query", "Image upload (find similar)"], index=0)

        if mode == "Text query":
            query = st.text_input("Query", placeholder="A person in a bright yellow raincoat")
            st.write("Try an example:")
            for ex in EXAMPLES:
                if st.button(ex, key=ex):
                    st.session_state["pending_query"] = ex
        else:
            uploaded = st.file_uploader(
                "Upload a product image",
                type=["jpg", "jpeg", "png", "webp"],
                help="Find visually similar products in the catalogue.",
            )
            query = None

        st.subheader("Retrieval")
        top_k = st.slider("Top-K", 1, 20, cfg.retrieval.top_k)
        use_rerank = st.checkbox("Cross-encoder re-ranker", value=cfg.retrieval.use_reranker)
        use_captions = st.checkbox("Use caption index", value=cfg.retrieval.use_captions)
        expand_q = st.checkbox(
            "Query expansion (5 variants)",
            value=getattr(cfg.retrieval, "expand_queries", True),
        )
        image_w = st.slider("Image weight", 0.0, 1.0, cfg.retrieval.image_weight, 0.05)
        cap_w = st.slider("Caption weight", 0.0, 1.0, cfg.retrieval.caption_weight, 0.05)
        rerank_w = st.slider(
            "Rerank weight",
            0.0,
            1.0,
            cfg.retrieval.rerank_weight,
            0.05,
        )
        scoring = st.selectbox(
            "Fusion",
            ["rrf", "weighted"],
            index=0 if getattr(cfg.retrieval, "scoring", "rrf") == "rrf" else 1,
        )
        attr_bonus = st.slider(
            "Attribute bonus",
            0.0,
            0.5,
            float(getattr(cfg.retrieval, "attribute_bonus", 0.20)),
            0.05,
            help="Multiplicative lift when the candidate's caption matches colour / garment / scene words from the query.",
        )
        show_caption = st.checkbox("Show image captions", value=True)
        show_breakdown = st.checkbox("Show score breakdown", value=True)

    if mode == "Text query":
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
                use_captions=use_captions,
                expand_queries=expand_q,
                image_weight=image_w,
                caption_weight=cap_w,
                rerank_weight=rerank_w,
                scoring=scoring,
                attribute_bonus=attr_bonus,
            ),
            captions=cfg.captions,
            rerank=cfg.rerank,
            log_level=cfg.log_level,
        )
        with st.spinner("Encoding query and searching..."):
            results = search(query, new_cfg, loaded=loaded, model=model)
        attrs = parse_query(query)
        axis_tags = query_axis_tags(attrs)
    else:
        if uploaded is None:
            st.info("Upload an image to find visually similar products.")
            return
        new_cfg = replace(cfg.retrieval, top_k=top_k)
        with st.spinner("Encoding uploaded image and searching..."):
            results = _image_to_image_search(
                uploaded, cfg, loaded, model, top_k,
                image_w, cap_w, use_rerank, use_captions, expand_q,
            )
        query = "(image upload)"
        attrs = None
        axis_tags = ["visual-similarity"]

    if not results:
        st.warning("No results.")
        return

    st.subheader(f"Top {len(results)} for:  {query}")
    if attrs is not None:
        st.write(
            f"Query axes: **{', '.join(axis_tags)}** | "
            f"colors: {[c for c in attrs.colors]} | "
            f"garments: {[g for g in attrs.garments]} | "
            f"scenes: {[s for s in attrs.scenes]}"
        )

    cols = st.columns(min(len(results), 5))
    for col, r in zip(cols * (len(results) // max(len(cols), 1) + 1), results):
        with col:
            data = _result_image(r.path)
            if data:
                st.image(data)
            st.markdown(f"**#{r.rank}**  score=`{r.score:.4f}`")
            if show_breakdown:
                st.caption(
                    f"img=`{r.image_score:.3f}` cap=`{r.caption_score:.3f}`"
                    + (f" rr=`{r.rerank_score:.3f}`" if r.rerank_score is not None else "")
                    + (f" rq=`{r.rerank_quality:.2f}`" if r.rerank_quality is not None else "")
                    + (f" ir=`{r.image_rank}`" if r.image_rank else "")
                    + (f" cr=`{r.caption_rank}`" if r.caption_rank else "")
                )
                if r.attribute_score:
                    st.caption(f"attr-match=`{r.attribute_score:.2f}`")
            if show_caption and r.caption:
                st.write(f"> {r.caption}")

    metrics = aggregate_metrics(
        [
            {"score": r.score, "path": str(r.path)} for r in results
        ],
        top_k=top_k,
    )
    with st.expander("Retrieval metrics"):
        st.json(metrics)

    with st.expander("Pipeline metadata"):
        st.json({
            "mode": mode,
            "query": query,
            "model": cfg.model.name,
            "pretrained": cfg.model.pretrained,
            "device": cfg.model.device,
            "image_index_ntotal": loaded.image_store.ntotal,
            "caption_index_ntotal": loaded.caption_store.ntotal if loaded.caption_store else 0,
            "captions_in_use": bool(loaded.caption_store) and use_captions,
            "reranker_in_use": bool(loaded.caption_store) and use_rerank,
            "fusion": scoring,
            "attribute_bonus": attr_bonus,
            "results_returned": len(results),
        })


if __name__ == "__main__":
    main()