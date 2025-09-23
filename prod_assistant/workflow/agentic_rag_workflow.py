from typing import Annotated, Sequence, TypedDict, Literal
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from prompt_library.prompts import PROMPT_REGISTRY, PromptType
from retriever.retrieval import Retriever
from utils.model_loader import ModelLoader
from langgraph.checkpoint.memory import MemorySaver

# tess - sep 17
from evaluation.ragas_eval import evaluate_context_precision, evaluate_response_relevancy



class AgenticRAG:
    """Agentic RAG pipeline using LangGraph."""

    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]

    def __init__(self):
        self.retriever_obj = Retriever()
        self.model_loader = ModelLoader()
        self.llm = self.model_loader.load_llm()
        self.checkpointer = MemorySaver()
        self.workflow = self._build_workflow()
        self.app = self.workflow.compile(checkpointer=self.checkpointer)

    # ---------- Helpers ----------
    def _format_docs(self, docs) -> str:
        if not docs:
            return "No relevant documents found."
        formatted_chunks = []
        for d in docs:
            meta = d.metadata or {}
            formatted = (
                f"Title: {meta.get('product_title', 'N/A')}\n"
                f"Price: {meta.get('price', 'N/A')}\n"
                f"Rating: {meta.get('rating', 'N/A')}\n"
                f"Reviews:\n{d.page_content.strip()}"
            )
            formatted_chunks.append(formatted)
        return "\n\n---\n\n".join(formatted_chunks)

    # ---------- Nodes ----------
    def _ai_assistant(self, state: AgentState):
        print("--- CALL ASSISTANT ---")
        messages = state["messages"]
        last_message = messages[-1].content

        if any(word in last_message.lower() for word in ["price", "review", "product"]):
            return {"messages": [HumanMessage(content="TOOL: retriever")]}
        else:
            prompt = ChatPromptTemplate.from_template(
                "You are a helpful assistant. Answer the user directly.\n\nQuestion: {question}\nAnswer:"
            )
            chain = prompt | self.llm | StrOutputParser()
            response = chain.invoke({"question": last_message})
            return {"messages": [HumanMessage(content=response)]}

    def _vector_retriever(self, state: AgentState):
        print("--- RETRIEVER ---")
        query = state["messages"][-1].content
        retriever = self.retriever_obj.load_retriever()
        docs = retriever.invoke(query)
        context = self._format_docs(docs)
        return {"messages": [HumanMessage(content=context)]}

    def _grade_documents(self, state: AgentState) -> Literal["generator", "rewriter"]:
        print("--- GRADER ---")
        question = state["messages"][0].content
        docs = state["messages"][-1].content

        prompt = PromptTemplate(
            template="""You are a grader. Question: {question}\nDocs: {docs}\n
            Are docs relevant to the question? Answer yes or no.""",
            input_variables=["question", "docs"],
        )
        chain = prompt | self.llm | StrOutputParser()
        score = chain.invoke({"question": question, "docs": docs})
        return "generator" if "yes" in score.lower() else "rewriter"

    # tess
    # def _generate(self, state: AgentState):  
    #     print("--- GENERATE ---")
    #     question = state["messages"][0].content
    #     docs = state["messages"][-1].content
    #     prompt = ChatPromptTemplate.from_template(
    #         PROMPT_REGISTRY[PromptType.PRODUCT_BOT].template
    #     )
    #     chain = prompt | self.llm | StrOutputParser()
    #     response = chain.invoke({"context": docs, "question": question})
    #     return {"messages": [HumanMessage(content=response)]}

    def _generate(self, state: AgentState):
        print("--- GENERATE ---")
        question = state["messages"][0].content

        # The message right before this node is the formatted context emitted by _vector_retriever
        # (If Assistant answered directly, this may not be a context block.)
        contexts_block = state["messages"][-1].content

        # Build the answer using your registered PRODUCT_BOT prompt
        prompt = ChatPromptTemplate.from_template(
            PROMPT_REGISTRY[PromptType.PRODUCT_BOT].template
        )
        chain = prompt | self.llm | StrOutputParser()
        answer = chain.invoke({"context": contexts_block, "question": question})

        # --- RAGAS scoring (only if we actually have retrieved contexts) ---
        # Your _format_docs used "\n\n---\n\n" between chunks; split it back into a list[str]
        retrieved_contexts = []
        if isinstance(contexts_block, str) and ("---" in contexts_block or "Title:" in contexts_block or "Reviews:" in contexts_block):
            retrieved_contexts = [c.strip() for c in contexts_block.split("\n\n---\n\n") if c.strip()]

        if retrieved_contexts:
            try:
                ctx_precision = evaluate_context_precision(question, answer, retrieved_contexts)
                resp_relevancy = evaluate_response_relevancy(question, answer, retrieved_contexts)

                # Log to server console
                print(f"[RAGAS] Context Precision: {ctx_precision:.3f} | Response Relevancy: {resp_relevancy:.3f}")

                # (Optional) append a tiny eval footer users can see
                answer = (
                    f"{answer}\n\n---\n"
                    f"[Eval] Context Precision: {ctx_precision:.2f} | Response Relevancy: {resp_relevancy:.2f}"
                )
            except Exception as e:
                print(f"[RAGAS] Evaluation error: {e}")

        return {"messages": [HumanMessage(content=answer)]}

    def _rewrite(self, state: AgentState):
        print("--- REWRITE ---")
        question = state["messages"][0].content
        new_q = self.llm.invoke(
            [HumanMessage(content=f"Rewrite the query to be clearer: {question}")]
        )
        return {"messages": [HumanMessage(content=new_q.content)]}

    # ---------- Build Workflow ----------
    def _build_workflow(self):
        workflow = StateGraph(self.AgentState)
        workflow.add_node("Assistant", self._ai_assistant)
        workflow.add_node("Retriever", self._vector_retriever)
        workflow.add_node("Generator", self._generate)
        workflow.add_node("Rewriter", self._rewrite)

        workflow.add_edge(START, "Assistant")
        workflow.add_conditional_edges(
            "Assistant",
            lambda state: "Retriever" if "TOOL" in state["messages"][-1].content else END,
            {"Retriever": "Retriever", END: END},
        )
        workflow.add_conditional_edges(
            "Retriever",
            self._grade_documents,
            {"generator": "Generator", "rewriter": "Rewriter"},
        )
        workflow.add_edge("Generator", END)
        workflow.add_edge("Rewriter", "Assistant")
        return workflow

    # ---------- Public Run ----------
    def run(self, query: str, thread_id: str = "default_thread") -> str:
        """Run the workflow for a given query and return the final answer."""
        result = self.app.invoke({"messages": [HumanMessage(content=query)]},
                                    config={"configurable": {"thread_id": thread_id}})
        return result["messages"][-1].content


if __name__ == "__main__":
    rag_agent = AgenticRAG()
    answer = rag_agent.run("What is the price of iPhone 15?")
    print("\nFinal Answer:\n", answer)
