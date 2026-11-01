import json
import re
import time
from pathlib import Path
from typing import Dict, List, TypedDict
import os
import requests
from dotenv import load_dotenv
from groq import RateLimitError
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from pydantic import BaseModel

from crew_node import run_research

load_dotenv()

MAX_LOOPS = 3
THRESHOLD = 0.8

MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
writer_llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.5)
critic_llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)

FORMATS = {
    "blog": dict(
        name="technical blog post", target=400, max=500, cap=600,
        rules="Start with an H1 title and a 2-sentence summary. Use markdown headers.",
        tone="technical blog",
        structure="H1 title, summary, headers",
    ),
    "linkedin": dict(
        name="LinkedIn post", target=200, max=250, cap=260,
        rules=("Open with a one-line hook. Use short paragraphs, no markdown headers, "
               "no tables, and end with 3-5 hashtags."),
        tone="professional but conversational LinkedIn",
        structure="strong hook, short paragraphs, hashtags at the end",
    ),
    "thread": dict(
        name="Twitter/X thread", target=250, max=320, cap=360,
        rules=("Write 6-8 numbered tweets (1/, 2/, ...). Each tweet must be under 280 "
               "characters. First tweet is a hook, last tweet is a takeaway. No headers."),
        tone="punchy Twitter thread",
        structure="numbered tweets, each under 280 characters, hook first, takeaway last",
    ),
}


class ContentState(TypedDict, total=False):
    topic: str
    format: str
    research: str
    sources: Dict[str, str]
    draft: str
    score: float
    feedback: List[str]
    unverified: List[str]
    loops: int
    flagged: bool
    output_path: str
    best_draft: str
    best_score: float
    best_unverified: List[str]


class Critique(BaseModel):
    feedback: List[str]
    traceable: float
    tone: float
    structure: float


def call_llm(llm, prompt, retries=6):
    """Call the LLM; on a per-minute rate limit, wait and retry."""
    for attempt in range(retries):
        try:
            return llm.invoke(prompt).content
        except RateLimitError as e:
            if "per day" in str(e) or "TPD" in str(e):
                raise RuntimeError(
                    "Groq daily token quota is used up. Wait for it to reset, or set "
                    "GROQ_MODEL in .env to another model."
                ) from e
            wait = 10 * (attempt + 1)
            print(f"[rate limit] waiting {wait}s (attempt {attempt + 1}/{retries})")
            time.sleep(wait)
    raise RuntimeError("Rate limit: retries exhausted")


def parse_critique(text: str) -> Critique:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("no JSON object found in critic output")
    return Critique.model_validate(json.loads(match.group(0)))


def ungrounded_numbers(draft: str, research: str) -> list:
    """Percentages in the draft that never appear in the research notes."""
    nums = set(re.findall(r"\d+(?:\.\d+)?\s?%", draft))
    research_clean = research.replace(" ", "")
    return sorted(n for n in nums if n.replace(" ", "") not in research_clean)


def fetch_sources(research: str, max_urls: int = 8) -> Dict[str, str]:
    """Download the pages whose URLs appear in the research brief."""
    urls = re.findall(r"https?://[^\s)\]>\"'*]+", research)
    urls = list(dict.fromkeys(u.rstrip(".,;") for u in urls))[:max_urls]
    headers = {
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
        "Accept-Language": "en-US,en;q=0.9",
    }
    pages = {}
    for url in urls:
        try:
            r = requests.get(url, headers=headers, timeout=10)
            r.raise_for_status()
            html = r.text[:1_500_000]
            text = re.sub(r"<(script|style).*?</\1>", " ", html, flags=re.S | re.I)
            text = re.sub(r"<[^>]+>", " ", text)
            pages[url] = re.sub(r"\s+", " ", text)
        except Exception as e:
            print(f"[factcheck] could not fetch {url}: {type(e).__name__}: {str(e)[:120]}")
    print(f"[factcheck] fetched {len(pages)}/{len(urls)} source pages")
    return pages


def unverified_numbers(draft: str, sources: Dict[str, str]) -> list:
    """Percentages in the draft that appear in none of the fetched source pages."""
    if not sources:
        return []
    corpus = " ".join(sources.values())
    nums = set(re.findall(r"\d+(?:\.\d+)?(?=\s?(?:%|percent))", draft))
    missing = []
    for n in sorted(nums):
        pattern = rf"(?<![\d.]){re.escape(n)}\s?(?:%|percent|per cent)"
        if not re.search(pattern, corpus, re.I):
            missing.append(f"{n}%")
    return missing


def research_node(state: ContentState) -> dict:
    for attempt in range(3):
        research = run_research(state["topic"])
        urls = set(re.findall(r"https?://[^\s)\]>\"'*]+", research))
        if len(urls) >= 2:
            break
        print(f"[research] attempt {attempt + 1} returned no usable sources, retrying")
    else:
        raise RuntimeError("Research failed: no usable source URLs after 3 attempts.")
    return {"research": research, "sources": fetch_sources(research), "loops": 0}


def draft_node(state: ContentState) -> dict:
    loops = state.get("loops", 0) + 1
    f = FORMATS[state.get("format", "blog")]
    prompt = (
        f"Write a {f['name']} of about {f['target']} words (never more than {f['max']}), "
        "using ONLY this research brief. Do not add claims that aren't in it.\n\n"
        f"TOPIC: {state['topic']}\n\nRESEARCH:\n{state['research']}\n\n"
        f"Format rules: {f['rules']} "
        "Never write 'Claim 1' or similar labels; name the actual source instead. "
        "Don't claim more than the research says."
    )
    if state.get("feedback"):
        fb = "\n".join(f"- {x}" for x in state["feedback"][:6])
        prompt += (
            f"\n\nPREVIOUS DRAFT:\n{state['draft']}\n\nEDITOR FEEDBACK:\n{fb}\n\n"
            f"Rewrite the post to fix these points. Target {f['target']} words, "
            f"HARD LIMIT {f['max']}. If you need space, remove or shorten content "
            "instead of adding. Do not add new sections. "
            "Delete any figure that is not in the research."
        )
    return {"draft": call_llm(writer_llm, prompt), "loops": loops}


def critique_node(state: ContentState) -> dict:
    fmt = state.get("format", "blog")
    f = FORMATS[fmt]
    prompt = (
        "You are a strict editor. First list EVERY problem in the draft as specific, "
        "actionable feedback items. Then score 0-1: traceable (every claim and number "
        f"appears in the research), tone ({f['tone']}), structure ({f['structure']}). "
        "If you listed any traceability problem, traceable must be 0.6 or lower.\n\n"
        f"RESEARCH:\n{state['research']}\n\nDRAFT:\n{state['draft']}\n\n"
        "Respond with ONLY a JSON object and no other text, in exactly this shape:\n"
        '{"feedback": ["problem 1", "problem 2"], "traceable": 0.0, "tone": 0.0, "structure": 0.0}'
    )

    c = None
    for attempt in range(3):
        raw = call_llm(critic_llm, prompt)
        try:
            c = parse_critique(raw)
            break
        except Exception as e:
            print(f"[critique] could not parse JSON (attempt {attempt + 1}): {e}")
    if c is None:
        c = Critique(feedback=["Critic failed to return valid JSON."],
                     traceable=0.5, tone=0.5, structure=0.5)

    clamp = lambda x: max(0.0, min(1.0, x))
    score = (clamp(c.traceable) + clamp(c.tone) + clamp(c.structure)) / 3
    feedback = list(c.feedback)

    bad_nums = ungrounded_numbers(state["draft"], state["research"])
    if bad_nums:
        feedback.insert(0, f"These figures are not in the research, remove them: {bad_nums}")
        score = min(score, 0.7)

    n_urls = len(set(re.findall(r"https?://[^\s)\]>\"'*]+", state["research"])))
    n_src = len(state.get("sources", {}))
    good_coverage = n_src >= max(2, (n_urls + 1) // 2)

    unverified = unverified_numbers(state["draft"], state.get("sources", {}))
    if unverified and good_coverage:
        feedback.insert(0, "These figures were not found in any fetched source page; "
                           f"remove them or soften the claim: {unverified}")
        score = min(score, 0.75)

    lines = state["draft"].splitlines()
    if fmt in ("linkedin", "thread"):
        has_table = any(l.strip().startswith("|") for l in lines)
        has_header = any(re.match(r"\s*#{1,6}\s", l) for l in lines)
        if has_table or has_header:
            feedback.insert(0, "Remove markdown tables and headers; use plain short "
                               "paragraphs (hashtags only at the end).")
            score = min(score, 0.75)
    if fmt == "thread":
        long_tweets = [l for l in lines if len(l) > 280]
        if long_tweets:
            feedback.insert(0, f"{len(long_tweets)} tweet(s) exceed 280 characters; shorten them.")
            score = min(score, 0.75)

    words = len(state["draft"].split())
    if words > f["cap"]:
        feedback.insert(0, f"Draft is {words} words; cut it to {f['target']}-{f['max']}.")
        score = min(score, 0.75)

    score = round(score, 2)

    print(f"[critique] sources fetched: {n_src}/{n_urls}")
    print(f"[critique] ungrounded: {bad_nums} | unverified: {unverified}")
    print(f"[critique] loop={state['loops']} score={score:.2f} words={words} issues={len(feedback)}")

    update = {"score": score, "feedback": feedback, "unverified": unverified}
    if score > state.get("best_score", -1):
        update.update(best_score=score, best_draft=state["draft"],
                      best_unverified=unverified)
    return update


def should_revise(state: ContentState) -> str:
    if state["score"] >= THRESHOLD:
        return "publish"
    return "draft" if state["loops"] < MAX_LOOPS else "publish"


def publish_node(state: ContentState) -> dict:
    draft = state.get("best_draft", state["draft"])
    score = state.get("best_score", state["score"])
    flagged = score < THRESHOLD
    fmt = state.get("format", "blog")
    slug = re.sub(r"[^a-z0-9]+", "-", state["topic"].lower()).strip("-")[:60]
    path = Path("output") / f"{slug}-{fmt}.md"
    path.parent.mkdir(exist_ok=True)
    note = ""
    if flagged:
        note = (f"> FLAGGED: best of {state['loops']} loops scored {score:.2f}, "
                f"below the {THRESHOLD} threshold. Needs human review.\n\n")
    path.write_text(note + draft, encoding="utf-8")
    return {"flagged": flagged, "output_path": str(path), "draft": draft,
            "score": score, "unverified": state.get("best_unverified", [])}


def build_app():
    g = StateGraph(ContentState)
    g.add_node("research", research_node)
    g.add_node("draft", draft_node)
    g.add_node("critique", critique_node)
    g.add_node("publish", publish_node)

    g.set_entry_point("research")
    g.add_edge("research", "draft")
    g.add_edge("draft", "critique")
    g.add_conditional_edges("critique", should_revise,
                            {"draft": "draft", "publish": "publish"})
    g.add_edge("publish", END)
    return g.compile()