import re
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient

async def main():
    client = MultiServerMCPClient ({
        "hybrid_search": {      # server name
            "command": "python",
            "args": [
                 r"C:\Users\tessd\LLMOPS\ecomm-prod-assistant\prod_assistant\mcp_servers\product_search_server.py"
        ],  # absolute path
        "transport": "stdio",
        }
    })

    # Discover tools
    tools = await client.get_tools()
    print("Available tools:", [t.name for t in tools])

    # Pick tools by name
    retriever_tool = next(t for t in tools if t.name == "get_product_info")
    web_tool = next(t for t in tools if t.name == "web_search")

    # ---- Step 1: Try retriever first ----------
    #query = "Samsung Galaxy S25 price"
    # query = "iPhone 15"
    query = "iPhone 17?"
    retriever_result = await retriever_tool.ainvoke({"query": query})
    print("\nRetriever Result:\n", retriever_result)

    def _query_number_present_in(text: str, query: str) -> bool:
        """
        Accept result only if any numeric token from query is present as a standalone number
        in the output. Since our server's formatted block includes a 'Title:' line,
        this is a reasonable proxy. (You could further restrict this to the Title line.)
        """
        nums = re.findall(r"\b\d+\b", query or "")
        if not nums:
            return True
        pattern = re.compile(
            r"(?<![A-Za-z0-9])(" + "|".join(map(re.escape, nums)) + r")(?![A-Za-z0-9])",
            flags=re.IGNORECASE,
        )
        return bool(pattern.search(text or ""))


    # ----- Step 2: Fallback to web search if retriever fails -------------
    # if not retriever_result.strip() or "No local results found." in retriever_result:
    #     print("\n No local results, falling back to web search...\n")
    #     web_result = await web_tool.ainvoke({"query": query})
    #     print("Web Search Result: \n", web_result)

    # after you print retriever_result:
    text = retriever_result if isinstance(retriever_result, str) else str(retriever_result)
    if (not text.strip()
        or "No local results found." in text
        or not _query_number_present_in(text, query)):
        print("\n No exact model match in local results, falling back to web search...\n")
        web_result = await web_tool.ainvoke({"query": query})
        print("Web Search Result: \n", web_result)

if __name__ == "__main__":
    asyncio.run(main())