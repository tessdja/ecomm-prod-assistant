from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from prompt_library.prompts import PROMPT_REGISTRY, PromptType
from retriever.retrieval_tess import Retriever
from utils.model_loader import ModelLoader
from evaluation.ragas_eval import evaluate_context_precision, evaluate_response_relevancy


retriever_obj = Retriever()
model_loader = ModelLoader()

def format_docs(docs) -> str:
    """Format retrieved documents into a structured text block for the prompt."""
    if not docs:
        return "No relevant documents found."
    
    formatted_chunks = []
    for d in docs:
        meta = d.metadata or {}
        formatted = (
            f"Title: {meta.get('product_title', 'N/A')}\n"
            f"Price: {meta.get('price', 'N/A')}\n"
            f"Rating: {meta.get('rating', 'N/A')}\n"
            f"Reviews: \n{d.page_content.strip()}"
        )
        
        formatted_chunks.append(formatted)
            
    return "\n\n---\n\n".join(formatted_chunks)

# Sunny merges all docs into one context item; Tess keeps one item per doc
# Sunny creates a single big string that contains all retrieved docs. With RAGAS’ context_precision, if that single blob
# contains any relevant material, precision tends to look like ~1.0 (because there’s only 1 “context” to judge).
# Tess returns one string per retrieved doc. If only half of those docs are actually relevant, 
# you’ll see precision around ~0.5—exactly what you observed.

def contexts_list(docs) -> list[str]:
    """Return one formatted string per doc (for RAGAS)."""
    items = []
    for d in docs:
        meta = d.metadata or {}
        items.append(
            f"Title: {meta.get('product_title', 'N/A')}\n"
            f"Price: {meta.get('price', 'N/A')}\n"
            f"Rating: {meta.get('rating', 'N/A')}\n"
            f"Reviews:\n{d.page_content.strip()}"
        )
    return items


def build_chain():
    """Build the RAG pipeline chain with retriever, prompt, LLM and parser."""
    retriever = retriever_obj.load_retriever()
    llm = model_loader.load_llm()
    prompt = ChatPromptTemplate.from_template(
        PROMPT_REGISTRY[PromptType.PRODUCT_BOT].template
    )

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def invoke_chain(query: str, debug: bool = False) -> str:
    """Run the chain with a user query and score with RAGAS."""
    chain = build_chain()  # build retriever|prompt|llm|parser chain
    # 1) Pull the docs once (same retriever as the chain uses)
    docs = retriever_obj.load_retriever().invoke(query)  # same call you use in debug
    if debug:
        print("\nRetrieved Documents:")
        print(format_docs(docs))
        print("\n-------\n")

    # 2) Run the chain to get the final answer
    answer = chain.invoke(query)  # original behavior
    # 3) Build RAGAS inputs and score
    try:
        retrieved_contexts = contexts_list(docs)  # one string per doc
        ctx_precision = evaluate_context_precision(query, answer, retrieved_contexts)
        resp_relevancy = evaluate_response_relevancy(query, answer, retrieved_contexts)
        print(f"[RAGAS] Context Precision: {ctx_precision:.3f} | Response Relevancy: {resp_relevancy:.3f}")

        # 4) (Optional) append a short footer for visibility
        answer = (
            f"{answer}\n\n---\n"
            f"[Eval] Context Precision: {ctx_precision:.2f} | Response Relevancy: {resp_relevancy:.2f}"
        )
    except Exception as e:
        print(f"[RAGAS] Evaluation error: {e}")

    # return answer

    return {
        "answer": answer,
        "metrics": {
            "context_precision": float(ctx_precision),
            "response_relevancy": float(resp_relevancy),
        }
    }


if __name__ == "__main__":
    try:
        answer = invoke_chain("Can you suggest good budget iPhone under 1,00,000 INR?")
        print("\n Assistant Answer: \n", answer)
    except Exception as e:
        import traceback
        print("Exception occurred:", str(e))
        traceback.print_exc()
