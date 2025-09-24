import importlib.metadata
packages = [
    "langchain",
    "python-dotenv",
    "python-multipart",
    "langchain_core",
    "streamlit",
    "fastapi",
    "selenium",
    "uvicorn",
    "beautifulsoup4",
    "jinja2",
    "lxml",
    "undetected-chromedriver",
    "langgraph",
    "structlog", 
    "html5lib",
    "langchain-astradb",
    "langchain-google-genai",
    "langchain-groq",
    "ragas",
    "mcp",
    "langchain-mcp-adapters",
    "ddgs"
]
for pkg in packages:
    try:
        version = importlib.metadata.version(pkg)
        print(f"{pkg}=={version}")
    except importlib.metadata.PackageNotFoundError:
        print(f"{pkg} (not installed)")
