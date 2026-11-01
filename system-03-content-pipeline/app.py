import time
from pathlib import Path

import re

import pandas as pd
import streamlit as st

import graph as pipeline
from graph import build_app, ungrounded_numbers

st.set_page_config(page_title="Content Pipeline", page_icon="✍️", layout="wide")

EXAMPLES = [
    "Why RAG systems need evaluation",
    "How vector databases speed up semantic search",
    "Guardrails for production LLM agents",
]
FORMAT_LABELS = {"Blog post": "blog", "LinkedIn post": "linkedin", "Twitter/X thread": "thread"}

if "history" not in st.session_state:
    st.session_state["history"] = []


def set_topic(t: str):
    st.session_state["topic"] = t


# ---------------- sidebar ----------------
with st.sidebar:
    st.header("⚙️ Settings")
    threshold = st.slider("Quality threshold", 0.5, 0.95, 0.8, 0.05,
                          help="Draft is published once the critic score reaches this.")
    max_loops = st.slider("Max revision loops", 1, 5, 3,
                          help="Hard cap, so the loop can never run forever.")
    st.caption("Higher threshold and more loops give better posts, "
               "but use more API tokens and time.")
    st.divider()
    st.subheader("💡 Example topics")
    for t in EXAMPLES:
        st.button(t, on_click=set_topic, args=(t,))


# ---------------- pipeline run ----------------
def run_pipeline(topic: str, fmt: str):
    pipeline.THRESHOLD = threshold
    pipeline.MAX_LOOPS = max_loops
    app = build_app()

    state, drafts, scores, feedbacks = {}, [], [], []
    start = time.time()
    try:
        with st.status("Running pipeline…", expanded=True) as status:
            for update in app.stream({"topic": topic, "format": fmt}, stream_mode="updates"):
                for node, out in update.items():
                    state.update(out)
                    t = time.time() - start
                    if node == "research":
                        status.write(f"🔎 Research crew finished, "
                                     f"{len(out['sources'])} source pages fetched ({t:.0f}s)")
                    elif node == "draft":
                        drafts.append(out["draft"])
                        status.write(f"📝 Draft v{out['loops']} written, "
                                     f"{len(out['draft'].split())} words ({t:.0f}s)")
                    elif node == "critique":
                        scores.append(out["score"])
                        feedbacks.append(out["feedback"])
                        status.write(f"🧐 Critique: score {out['score']:.2f}, "
                                     f"{len(out['feedback'])} issues ({t:.0f}s)")
                    elif node == "publish":
                        status.write(f"📤 Published ({t:.0f}s)")
            status.update(label=f"Done in {time.time() - start:.0f}s",
                          state="complete", expanded=False)
    except Exception as e:
        st.error(f"Pipeline failed: {e}")
        return

    elapsed = time.time() - start
    st.session_state["result"] = dict(
        topic=topic, state=state, drafts=drafts, scores=scores,
        feedbacks=feedbacks, elapsed=elapsed, threshold=threshold,
    )
    st.session_state["history"].append({
        "Topic": topic,
        "Format": fmt,
        "Score": round(state["score"], 2),
        "Loops": state["loops"],
        "Words": len(state["draft"].split()),
        "Flagged": state["flagged"],
        "Seconds": round(elapsed),
    })


# ---------------- result display ----------------
def show_result():
    r = st.session_state.get("result")
    if not r:
        return
    s, drafts, scores = r["state"], r["drafts"], r["scores"]

    st.subheader(f"Result: {r['topic']}")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Loops", s["loops"])
    c2.metric("Final score", f"{s['score']:.2f}")
    c3.metric("Words", len(s["draft"].split()))
    c4.metric("Time", f"{r['elapsed']:.0f}s")
    c5.metric("Status", "⚠️ Flagged" if s["flagged"] else "✅ Passed")

    if s["flagged"]:
        st.warning("Loop limit reached below the threshold. This post needs human review.")
    else:
        st.success("Passed the quality gate.")

    bad = ungrounded_numbers(s["draft"], s["research"])
    if bad:
        st.warning(f"Figures in the post that are not in the research: {bad}")

    unv = s.get("unverified", [])
    n_src = len(s.get("sources", {}))
    n_urls = len(set(re.findall(r"https?://[^\s)\]>\"'*]+", s["research"])))
    has_pct = re.search(r"\d+(?:\.\d+)?\s?(?:%|percent)", s["draft"])
    if n_src == 0:
        st.info("Fact-check skipped: no source pages could be fetched.")
    elif not has_pct:
        st.info("Fact-check: the post contains no percentages, so there was nothing to check.")
    elif unv and n_src < max(2, n_urls // 2):
        st.warning(f"Weak fact-check: only {n_src} of {n_urls} source pages could be fetched, "
                   f"so these figures are unconfirmed, not necessarily wrong: {unv}")
    elif unv:
        st.warning(f"Fact-check: not found in any of the {n_src} fetched pages: {unv}")
    else:
        st.success(f"Fact-check: every percentage was found in the {n_src} of {n_urls} fetched pages.")

    # unv = s.get("unverified", [])
    # n_src = len(s.get("sources", {}))
    # has_pct = re.search(r"\d+(?:\.\d+)?\s?(?:%|percent)", s["draft"])
    # if n_src == 0:
    #     st.info("Fact-check skipped: no source pages could be fetched.")
    # elif not has_pct:
    #     st.info("Fact-check: the post contains no percentages, so there was nothing to check.")
    # elif unv:
    #     st.warning(f"Fact-check: not found in any of the {n_src} fetched source pages: {unv}")
    # else:
    #     st.success(f"Fact-check: every percentage was found in the {n_src} fetched source pages.")

    # unv = s.get("unverified", [])
    # n_src = len(s.get("sources", {}))
    # if n_src == 0:
    #     st.info("Fact-check skipped: no source pages could be fetched.")
    # elif unv:
    #     st.warning(f"Fact-check: not found in any of the {n_src} fetched source pages: {unv}")
    # else:
    #     st.success(f"Fact-check: every percentage was found in the {n_src} fetched source pages.")

    st.info("The critic checks the post against the research notes, and the fact-check "
            "only covers percentages. Verify any statistic or paper before publishing.")

    t1, t2, t3, t4 = st.tabs(["📄 Final post", "🕘 Draft versions",
                              "📊 Loop analysis", "🔎 Research brief"])

    with t1:
        st.markdown(s["draft"])
        try:
            final_text = Path(s["output_path"]).read_text(encoding="utf-8")
        except Exception:
            final_text = s["draft"]
        st.download_button("⬇️ Download .md", final_text,
                           file_name=Path(s["output_path"]).name)

    with t2:
        if len(drafts) > 1:
            v = st.select_slider("Version", options=list(range(1, len(drafts) + 1)),
                                 value=len(drafts))
        else:
            v = 1
        st.caption(f"Loop {v}: score {scores[v - 1]:.2f}, "
                   f"{len(drafts[v - 1].split())} words")
        st.markdown(drafts[v - 1])

    with t3:
        n = len(scores)
        idx = list(range(1, n + 1))
        col_a, col_b = st.columns(2)
        with col_a:
            st.caption("Critic score per loop vs threshold")
            st.line_chart(pd.DataFrame(
                {"score": scores, "threshold": [r["threshold"]] * n}, index=idx))
        with col_b:
            st.caption("Words per draft")
            st.bar_chart(pd.DataFrame(
                {"words": [len(d.split()) for d in drafts]}, index=idx))
        for i, fb in enumerate(r["feedbacks"], start=1):
            with st.expander(f"Loop {i}: score {scores[i - 1]:.2f}, {len(fb)} issues"):
                for f in fb:
                    st.markdown(f"- {f}")

    with t4:
        st.markdown(s["research"])
        st.download_button("⬇️ Download research brief", s["research"],
                           file_name="research_brief.md")


# ---------------- page ----------------
st.title("✍️ Autonomous Content Pipeline")
st.caption("CrewAI research crew → LangGraph draft ↔ critique loop → fact-check → publish")

tab_gen, tab_hist, tab_how = st.tabs(["🚀 Generate", "🗂 History", "🧭 How it works"])

with tab_gen:
    topic = st.text_input("Topic", key="topic",
                          placeholder="Why RAG systems need evaluation")
    fmt_label = st.radio("Format", list(FORMAT_LABELS), horizontal=True)
    if st.button("Generate", type="primary"):
        if not topic.strip():
            st.warning("Please enter a topic first.")
        else:
            run_pipeline(topic.strip(), FORMAT_LABELS[fmt_label])
    show_result()

with tab_hist:
    st.subheader("This session")
    if st.session_state["history"]:
        st.dataframe(pd.DataFrame(st.session_state["history"]), hide_index=True)
    else:
        st.caption("No runs yet.")

    st.subheader("Saved posts")
    files = sorted(Path("output").glob("*.md"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    if files:
        pick = st.selectbox("Open a saved post", [f.name for f in files])
        st.markdown(Path("output", pick).read_text(encoding="utf-8"))
    else:
        st.caption("No saved posts yet.")

with tab_how:
    st.graphviz_chart(r'''
    digraph {
      rankdir=LR; bgcolor="transparent";
      node [shape=box, style="rounded,filled", fillcolor="#1f2937",
            fontcolor="white", color="#4b5563"];
      edge [color="#9ca3af", fontcolor="#9ca3af"];
      topic [label="Topic + format"];
      research [label="Research crew\n(CrewAI, 3 agents)\n+ fetch source pages"];
      draft [label="Draft"];
      critique [label="Critique\n(LLM + code checks\n+ fact-check)"];
      publish [label="Publish\n(.md file)"];
      topic -> research -> draft -> critique;
      critique -> draft [label="score < threshold\nand loops < max"];
      critique -> publish [label="pass, or loops used up"];
    }
    ''')
    st.markdown(
        "**Research:** three CrewAI agents (librarian, analyst, writer) search the web "
        "and write a brief. The source pages named in it are then downloaded.  \n"
        "**Draft ↔ Critique:** a LangGraph loop. The critic's score also has code-based "
        "checks (unsupported numbers, numbers missing from the fetched sources, length), "
        "because an LLM judge alone was too lenient.  \n"
        "**Formats:** the same pipeline writes a blog post, a LinkedIn post or a Twitter/X "
        "thread, each with its own length and structure rules.  \n"
        "**Publish:** saved as Markdown, with a FLAGGED note if the quality gate was not met."
    )