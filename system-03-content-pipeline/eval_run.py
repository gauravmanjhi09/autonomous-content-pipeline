import csv
import re
import sys
import time
from pathlib import Path

from graph import build_app

TOPICS = [
    "Why RAG systems need evaluation",
    "How vector databases speed up semantic search",
    "Guardrails for production LLM agents",
    "Prompt caching and why it cuts LLM costs",
    "Fine-tuning vs retrieval: when to use which",
    "How embeddings represent meaning",
    "Why LLM agents need structured outputs",
    "Chunking strategies for RAG pipelines",
    "Monitoring LLM applications in production",
    "Reducing hallucinations in LLM apps",
]
FORMAT = "blog"          
PAUSE_SECONDS = 15
OUT = Path("eval_results.csv")
FIELDS = ["topic", "format", "loops", "first_score", "final_score", "flagged",
          "words", "sources_fetched", "sources_total", "has_percentages",
          "unverified_count", "seconds"]


def done_topics():
    if not OUT.exists():
        return set()
    with OUT.open(encoding="utf-8", newline="") as f:
        return {(r["topic"], r["format"]) for r in csv.DictReader(f)}


def run_one(app, topic):
    start = time.time()
    scores, state = [], {}
    for update in app.stream({"topic": topic, "format": FORMAT}, stream_mode="updates"):
        for node, out in update.items():
            state.update(out)
            if node == "critique":
                scores.append(out["score"])
    draft = state["draft"]
    return {
        "topic": topic,
        "format": FORMAT,
        "loops": state["loops"],
        "first_score": round(scores[0], 3),
        "final_score": round(state["score"], 3),
        "flagged": state["flagged"],
        "words": len(draft.split()),
        "sources_fetched": len(state.get("sources", {})),
        "sources_total": len(set(re.findall(r"https?://[^\s)\]>\"'*]+", state["research"]))),
        "has_percentages": bool(re.search(r"\d+(?:\.\d+)?\s?(?:%|percent)", draft)),
        "unverified_count": len(state.get("unverified", [])),
        "seconds": round(time.time() - start),
    }


def summary():
    if not OUT.exists():
        print("No results yet.")
        return
    with OUT.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    n = len(rows)
    if n == 0:
        print("No results yet.")
        return
    avg = lambda key: sum(float(r[key]) for r in rows) / n
    passed = sum(r["flagged"] == "False" for r in rows)
    print(f"\n=== Evaluation summary ({n} topics) ===")
    print(f"Published without a flag : {passed}/{n} ({100 * passed / n:.0f}%)")
    print(f"Average revision loops   : {avg('loops'):.1f}")
    print(f"Average first-draft score: {avg('first_score'):.2f}")
    print(f"Average final score      : {avg('final_score'):.2f}")
    print(f"Average words            : {avg('words'):.0f}")
    print(f"Average time per run     : {avg('seconds'):.0f}s")


if __name__ == "__main__":
    if "--summary" in sys.argv:
        summary()
        sys.exit()

    app = build_app()
    finished = done_topics()
    new_file = not OUT.exists()
    with OUT.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        for topic in TOPICS:
            if (topic, FORMAT) in finished:
                print(f"[skip] already done: {topic}")
                continue
            print(f"\n[eval] running: {topic}")
            try:
                row = run_one(app, topic)
            except Exception as e:
                msg = str(e)
                if "per day" in msg or "TPD" in msg or "daily" in msg:
                    print("[eval] Daily Groq quota reached. Run this script again later; "
                          "finished topics are saved and will be skipped.")
                    break
                print(f"[eval] failed on '{topic}': {type(e).__name__}: {msg[:200]}")
                continue
            writer.writerow(row)
            f.flush()
            print(f"[eval] done: score {row['first_score']} -> {row['final_score']}, "
                  f"loops={row['loops']}, flagged={row['flagged']}")
            time.sleep(PAUSE_SECONDS)
    summary()