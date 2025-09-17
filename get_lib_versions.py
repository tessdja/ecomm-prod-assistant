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
    "structlog"
]
for pkg in packages:
    try:
        version = importlib.metadata.version(pkg)
        print(f"{pkg}=={version}")
    except importlib.metadata.PackageNotFoundError:
        print(f"{pkg} (not installed)")
