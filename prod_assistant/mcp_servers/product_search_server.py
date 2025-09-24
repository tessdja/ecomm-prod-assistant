import re
import textwrap
from mcp.server.fastmcp import FastMCP
from retriever.retrieval import Retriever
# from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

# Initialize MCP server
mcp = FastMCP("hybrid_search")

# Load retriever once
retriever_obj = Retriever()
retriever = retriever_obj.load_retriever()

# Langchain DuckDuckGo tool
# duckduckgo = DuckDuckGoSearchRun()  # instantitate
# ddg_results = DuckDuckGoSearchResults()  # returns a list[dict] on run/invoke
ddg = DuckDuckGoSearchAPIWrapper(region="wt-wt", time="y", max_results=10)

# ---------- Helpers -------------
def format_docs(docs) -> str:
    """Format retreiver docs into readable context"""
    if not docs:
        return ""
    formatted_chunks = []
    for d in docs:
        meta = d.metadata or {}
        formatted = (
            f"Title: {meta.get('product_title', 'N/A')}\n"
            f"Price: {meta.get('price', 'N/A')}\n"
            f"Rating: {meta.get('rating', 'N/A')}\n"
            f"Review: \n{d.page_content.strip()}"
        )
        formatted_chunks.append(formatted)
    return "\n\n---\n\n".join(formatted_chunks)

def _number_filter(docs, query: str):
    """
    Keep docs only if:
      1) Any product keyword from query (e.g., 'iphone', 'samsung') is present in the TITLE, and
      2) Any numeric token from query (e.g., '17') appears as a standalone number in the TITLE.
    If the query has no digits, we only check the keyword condition.
    """
    q = (query or "").lower()

    # crude brand/product keywords from the query (expand if you want)
    # You can also parse tokens and keep only alphabetic ones
    tokens = [t for t in re.findall(r"[a-zA-Z]+", q)]
    keywords = set(tokens)

    # numeric tokens from the query (model numbers like 17, 256, etc.)
    nums = re.findall(r"\b\d+\b", q)
    want_nums = set(nums)

    def has_keyword_in_title(title: str) -> bool:
        t = (title or "").lower()
        # require at least one query word to be in title; if no words, allow
        return True if not keywords else any(w in t for w in keywords)

    # compile a strict numeric pattern: numbers must be standalone (not A17, not 117)
    num_pattern = None
    if want_nums:
        num_pattern = re.compile(
            r"(?<![A-Za-z0-9])(" + "|".join(map(re.escape, want_nums)) + r")(?![A-Za-z0-9])",
            flags=re.IGNORECASE,
        )

    filtered = []
    for d in docs:
        title = ((d.metadata or {}).get("product_title", "")) or ""
        if not has_keyword_in_title(title):
            continue
        if num_pattern:
            if not num_pattern.search(title):
                continue
        filtered.append(d)

    return filtered

def _fmt_ddg_results(items, k: int = 5) -> str:
    """
    Pretty-print the top-k DuckDuckGo results.
    Each item is expected to be a dict with keys: title, link, snippet/description.
    """
    if not items:
        return "No web results."

    lines = []
    for i, it in enumerate(items[:k], start=1):
        title = (it.get("title") or it.get("heading") or "(no title)").strip()
        link = (it.get("link") or it.get("href") or "").strip()
        snippet = (it.get("snippet") or it.get("body") or it.get("description") or "").strip()

        # Wrap the snippet a bit for readability in your console
        snippet_wrapped = textwrap.fill(snippet, width=90) if snippet else ""

        block = f"{i}. {title}\n   {link}"
        if snippet_wrapped:
            block += f"\n   {snippet_wrapped}"
        lines.append(block)

    return "\n\n".join(lines)


# ----------- MCP Tools -------------------------
@mcp.tool()
async def get_product_info(query: str) -> str:
    """Retrieve product information for a given query from local retriever."""
    try:
        docs = retriever.invoke(query)
        docs = _number_filter(docs, query)
        context = format_docs(docs)
        if not context.strip():
            return "No local results found."
        return context
    except Exception as e:
        return f"Error retrieving product info {str(e)}"
    
@mcp.tool()
async def web_search(query: str) -> str:
    """Search the web using DuckDuckGo if retriever has no results."""
    try:
        # either of these are fine
        # return duckduckgo.run(query)
        # return duckduckgo.invoke(query)   # Langchain standard invoke
        
        # `invoke` or `run` both work; `invoke` is more LC-standard
        results = ddg.results(query, max_results=5)   # -> list[dict]
        return _fmt_ddg_results(results, k=5)
    except Exception as e:
        return f"Error during web search: {str(e)}"
    
# --------- Run Server ----------------
if __name__ == "__main__":
    mcp.run(transport="stdio")
    