"""
End-to-end pipeline: loads data, builds embeddings + FAISS index,
sets up the summarizer, keyword extractor and LLM agent, then runs a
sample query through the agent.

Run with:  python -m src.main
"""

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from . import config
from .agent_tools import build_all_tools
from .data_loader import load_papers_dataframe
from .embeddings import generate_embeddings, load_embedding_model
from .keyword_extractor import load_keyword_model
from .llm import load_llm
from .search_index import build_or_load_index, print_search_results
from .summarizer import load_summarizer


def build_pipeline():
    """Load data and models, and build the search index. Returns all shared resources."""
    df = load_papers_dataframe()

    model = load_embedding_model()
    embeddings = generate_embeddings(model, df["paper_text"].tolist())
    index = build_or_load_index(embeddings)

    summarizer = load_summarizer()
    kw_model = load_keyword_model(model)
    llm = load_llm()

    tools = build_all_tools(model, index, df, summarizer, kw_model, llm)

    return {
        "df": df,
        "model": model,
        "index": index,
        "summarizer": summarizer,
        "kw_model": kw_model,
        "llm": llm,
        "tools": tools,
    }


def run_sample_query(resources, user_query: str):
    """Run a sample query through the LLM + tools, mirroring a manual tool-call loop."""
    llm = resources["llm"]
    tools = resources["tools"]

    llm_with_tools = llm.bind_tools(tools)
    response = llm_with_tools.invoke(user_query)

    if not response.tool_calls:
        print(response.content)
        return

    tool_call = response.tool_calls[0]
    tool_name = tool_call["name"]
    tool_args = tool_call["args"]

    tool_by_name = {t.name: t for t in tools}
    tool_result = tool_by_name[tool_name].invoke(tool_args)

    tool_message = ToolMessage(content=tool_result, tool_call_id=tool_call["id"])

    final_response = llm.invoke(
        [
            SystemMessage(
                content="""
You are a helpful AI assistant.

Rules:
1. Always use the tool output.
2. Never ignore tool results.
3. Present the complete tool output.
4. Add a short explanation after the tool output if necessary.
"""
            ),
            HumanMessage(content=user_query),
            response,
            tool_message,
        ]
    )

    print(final_response.content)


def run_agent_demo(resources, user_query: str):
    """Run a query through the full LangChain agent (multi-step tool use)."""
    agent = create_agent(model=resources["llm"], tools=resources["tools"])

    response = agent.invoke(
        {"messages": [{"role": "user", "content": user_query}]}
    )
    return response


if __name__ == "__main__":
    resources = build_pipeline()

    # Quick sanity check: plain similarity search
    print_search_results(
        "deep learning for medical image analysis",
        resources["model"],
        resources["index"],
        resources["df"],
    )

    # Manual single tool-call flow
    run_sample_query(
        resources,
        "Extract the top 5 keywords from Deep Learning for Medical Image Reconstruction.",
    )

    # Full agent flow
    agent_response = run_agent_demo(
        resources, "Find the top 3 research papers on Vision Transformer."
    )
    print(agent_response)
