import json
import re
from pathlib import Path
from typing import List, TypedDict

import time
from groq import RateLimitError

import os
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

from crew_node import run_research

load_dotenv()

MAX_LOOPS = 3
THRESHOLD = 0.8

writer_llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.5)
critic_llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)


class ContentState(TypedDict, total=False):
    topic: str
    research: str
    draft: str
    score: float
    feedback: List[str]
    loops: int
    flagged: bool
    output_path: str


# class Critique(BaseModel):
#     traceable: float   # 0-1: every claim traceable to the research
#     tone: float        # 0-1: matches a technical blog
#     structure: float   # 0-1: clear structure with headers
#     feedback: List[str]  # specific, actionable fixes

class Critique(BaseModel):
    feedback: List[str]  # list EVERY problem first
    traceable: float
    tone: float
    structure: float

def research_node(state: ContentState) -> dict:
    return {"research": run_research(state["topic"]), "loops": 0}


def draft_node(state: ContentState) -> dict:
    loops = state.get("loops", 0) + 1
    prompt = (
        "Write a technical blog post of about 400 words (never more than 500) with markdown headers, using ONLY "
        f"this research brief. Do not add claims that aren't in it.\n\n"
        f"TOPIC: {state['topic']}\n\nRESEARCH:\n{state['research']}"
                "\n\nRules: start with an H1 title and a 2-sentence summary. Never write 'Claim 1' "
        "or similar labels; name the actual source instead. Stay near 400 words and never above 500.  "
        "Don't claim more than the research says."
    )
    if state.get("feedback"):
        fb = "\n".join(f"- {f}" for f in state["feedback"][:6])
        prompt += (
            f"\n\nPREVIOUS DRAFT:\n{state['draft']}\n\nEDITOR FEEDBACK:\n{fb}\n\n"
            "Rewrite the post to fix these points. Target 400 words, HARD LIMIT 500. "
            "If you need space, remove or shorten content instead of adding. "
            "Do not add new sections. Delete any figure that is not in the research."
        )
    
    # if state.get("feedback"):
    #     fb = "\n".join(f"- {f}" for f in state["feedback"])
    #     prompt += (
    #         f"\n\nPREVIOUS DRAFT:\n{state['draft']}\n\nCRITIC FEEDBACK:\n{fb}\n\n"
    #         "Revise the previous draft. Address every feedback point; don't rewrite from scratch."
    #     )
    draft = call_llm(writer_llm, prompt)
    return {"draft": draft, "loops": loops}

def ungrounded_numbers(draft: str, research: str) -> list:
    """Percentages in the draft that never appear in the research."""
    nums = set(re.findall(r"\d+(?:\.\d+)?\s?%", draft))
    research_clean = research.replace(" ", "")
    return sorted(n for n in nums if n.replace(" ", "") not in research_clean)

# def critique_node(state: ContentState) -> dict:
#     judge = critic_llm.with_structured_output(Critique)
#     c = judge.invoke(
#         "Score this draft from 0 to 1 on: traceable (every claim appears in the research), "
#         "tone (technical blog), structure (clear headers). Be strict. Give specific, "
#         "actionable feedback items.\n\n"
#         f"RESEARCH:\n{state['research']}\n\nDRAFT:\n{state['draft']}"
#     )
#     clamp = lambda x: max(0.0, min(1.0, x))
#     score = (clamp(c.traceable) + clamp(c.tone) + clamp(c.structure)) / 3
#     print(f"[critique] loop={state['loops']} score={score:.2f} feedback={c.feedback}")
#     return {"score": score, "feedback": c.feedback}

def parse_critique(text: str) -> Critique:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("no JSON object found in critic output")
    return Critique.model_validate(json.loads(match.group(0)))

def call_llm(llm, prompt, retries=6):
    """Call the LLM; on a rate limit, wait and try again."""
    for attempt in range(retries):
        try:
            return llm.invoke(prompt).content
        except RateLimitError:
            wait = 10 * (attempt + 1)
            print(f"[rate limit] waiting {wait}s (attempt {attempt + 1}/{retries})")
            time.sleep(wait)
    raise RuntimeError("Rate limit: retries exhausted")

def critique_node(state: ContentState) -> dict:
    prompt = (
        "You are a strict editor. First list EVERY problem in the draft as specific, "
        "actionable feedback items. Then score 0-1: traceable (every claim and number "
        "appears in the research), tone (technical blog), structure (H1 title, summary, "
        "headers). If you listed any traceability problem, traceable must be 0.6 or lower.\n\n"
        f"RESEARCH:\n{state['research']}\n\nDRAFT:\n{state['draft']}\n\n"
    ) + (
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
        feedback.append(f"These figures are not in the research, remove or source them: {bad_nums}")
        score = min(score, 0.7)

    words = len(state["draft"].split())
    if words > 600:
        feedback.insert(0, f"Draft is {words} words; cut it to 400-500.")
        score = min(score, 0.75)

    print(f"[critique] ungrounded numbers: {bad_nums}")

    print(f"[critique] loop={state['loops']} score={score:.2f} words={words} issues={len(feedback)}")
    return {"score": score, "feedback": feedback}



# def critique_node(state: ContentState) -> dict:
#     judge = critic_llm.with_structured_output(Critique)
#     c = judge.invoke(
#         "You are a strict editor. First list EVERY problem in the draft as specific, "
#         "actionable feedback items. Then score 0-1: traceable (every claim and number "
#         "appears in the research), tone (technical blog), structure (H1 title, summary, "
#         "headers). If you listed any traceability problem, traceable must be 0.6 or lower.\n\n"
#         f"RESEARCH:\n{state['research']}\n\nDRAFT:\n{state['draft']}"
#     )
#     clamp = lambda x: max(0.0, min(1.0, x))
#     score = (clamp(c.traceable) + clamp(c.tone) + clamp(c.structure)) / 3
#     feedback = list(c.feedback)

#     bad_nums = ungrounded_numbers(state["draft"], state["research"])
#     if bad_nums:
#         feedback.append(f"These figures are not in the research, remove or source them: {bad_nums}")
#         score = min(score, 0.7)

#     words = len(state["draft"].split())
#     if words > 650:
#         feedback.append(f"Draft is {words} words; cut it to about 500.")
#         score = min(score, 0.75)

#     print(f"[critique] loop={state['loops']} score={score:.2f} words={words} issues={len(feedback)}")
#     return {"score": score, "feedback": feedback}


def should_revise(state: ContentState) -> str:
    if state["score"] >= THRESHOLD:
        return "publish"
    return "draft" if state["loops"] < MAX_LOOPS else "publish"


def publish_node(state: ContentState) -> dict:
    flagged = state["score"] < THRESHOLD
    slug = re.sub(r"[^a-z0-9]+", "-", state["topic"].lower()).strip("-")[:60]
    path = Path("output") / f"{slug}.md"
    path.parent.mkdir(exist_ok=True)
    note = ""
    if flagged:
        note = (f"> FLAGGED: published after {MAX_LOOPS} loops with score "
                f"{state['score']:.2f}. Needs human review.\n\n")
    path.write_text(note + state["draft"], encoding="utf-8")
    return {"flagged": flagged, "output_path": str(path)}


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