from dotenv import load_dotenv
# from duckduckgo_search import DDGS
from ddgs import DDGS
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
import os

load_dotenv()

llm = LLM(
    model="openai/openai/gpt-oss-120b",
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.3,
)

# @tool("web_search")
# def web_search(query: str) -> str:
#     """Search the web. Input: a search query string, e.g. 'agentic AI trends 2026'.
#     Returns title, URL and snippet for the top results."""
#     with DDGS() as ddgs:
#         results = ddgs.text(query, max_results=6)
#     return "\n\n".join(f"{r['title']}\n{r['href']}\n{r['body']}" for r in results)

@tool("web_search")
def web_search(query: str) -> str:
    """Search the web. Input: a search query string, e.g. 'agentic AI trends 2026'.
    Returns title, URL and snippet for the top results."""
    try:
        results = DDGS().text(query, max_results=6)
    except Exception as e:
        return f"Search failed: {e}. Try a shorter, different query."
    if not results:
        return "No results found. Try a different query."
    return "\n\n".join(f"{r['title']}\n{r['href']}\n{r['body']}" for r in results)


def run_research(topic: str) -> str:
    searcher = Agent(
        max_iter=6,
        role="Senior Research Librarian",
        goal="Find 5-6 credible, recent sources on {topic}; return title, URL, 1-line note.",
        backstory="You distrust SEO content farms and prefer official docs, papers, first-party blogs.",
        tools=[web_search], llm=llm, allow_delegation=False,
    )
    analyst = Agent(
        role="Critical Research Analyst",
        goal="Extract the 3-5 strongest claims, each with evidence and source URL.",
        backstory="You flag unsupported claims and never merge two sources' claims without saying so.",
        llm=llm, allow_delegation=False,
    )
    writer = Agent(
        role="Technical Report Writer",
        goal="Turn the analyst's claims into a structured research brief.",
        backstory="You write for busy engineers. Open with the answer.",
        llm=llm, allow_delegation=False,
    )
    t1 = Task(description="Research: {topic}. Use web_search.",
              expected_output="5-6 sources: title, URL, relevance note.", agent=searcher)
    t2 = Task(description="Extract strongest claims on {topic}.",
              expected_output="3-5 claims with evidence and source URL.",
              agent=analyst, context=[t1])
    t3 = Task(description="Write a research brief on {topic} using only the claims.",
              expected_output="Short summary, headed sections, sources list (Markdown).",
              agent=writer, context=[t2])

    crew = Crew(agents=[searcher, analyst, writer], tasks=[t1, t2, t3],
                process=Process.sequential)
    result = crew.kickoff(inputs={"topic": topic})
    return getattr(result, "raw", str(result))