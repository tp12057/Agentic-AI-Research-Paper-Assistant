"""
LangChain tool definitions that wrap the search, summarization, keyword
extraction, and paper comparison functionality for use by the agent.

Each tool is built by a factory function so it can close over the shared
model / index / dataframe / summarizer / kw_model / llm objects instead of
relying on module-level globals.
"""

import faiss
from langchain_core.tools import tool


def build_search_and_summarize_tool(model, index, df, summarizer):
    """Create the 'search_and_summarize' tool bound to the given resources."""

    @tool
    def search_and_summarize(query: str, k: int = 5):
        """
        Search research papers from the FAISS database,
        retrieve the top-k most similar papers,
        summarize each paper using BART,
        and return the results as structured data.
        """
        query_embedding = model.encode([query])
        faiss.normalize_L2(query_embedding)

        distances, indices = index.search(query_embedding, k)

        papers = []
        for rank, (score, idx) in enumerate(zip(distances[0], indices[0]), start=1):
            paper = df.iloc[idx]

            summary = summarizer(
                paper["abstract"],
                max_length=120,
                min_length=40,
                do_sample=False,
            )[0]["summary_text"]

            papers.append(
                {
                    "rank": rank,
                    "similarity": round(float(score), 4),
                    "title": paper["title"],
                    "abstract": paper["abstract"],
                    "summary": summary,
                }
            )

        return papers

    return search_and_summarize


def build_extract_keywords_tool(kw_model):
    """Create the 'extract_keywords' tool bound to the given KeyBERT model."""

    @tool
    def extract_keywords(text: str, top_n: int = 5) -> str:
        """
        Extract the most important keywords from the given text
        using the KeyBERT model.
        """
        keywords = kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 2),
            stop_words="english",
            top_n=top_n,
        )

        result = "Top Keywords:\n\n"
        for rank, (keyword, score) in enumerate(keywords, start=1):
            result += f"{rank}. {keyword} (Relevance Score: {round(score, 4)})\n"

        return result

    return extract_keywords


def build_compare_papers_tool(model, index, df, llm):
    """Create the 'compare_papers' tool bound to the given resources."""

    @tool
    def compare_papers(paper1: str, paper2: str) -> str:
        """
        Compare two research papers based on their abstracts.
        The tool retrieves the closest matching papers from the FAISS database
        and uses the LLM to compare them.
        """
        embedding1 = model.encode([paper1])
        faiss.normalize_L2(embedding1)
        _, indices1 = index.search(embedding1, 1)
        first_paper = df.iloc[indices1[0][0]]

        embedding2 = model.encode([paper2])
        faiss.normalize_L2(embedding2)
        _, indices2 = index.search(embedding2, 1)
        second_paper = df.iloc[indices2[0][0]]

        comparison_prompt = f"""
Compare the following two research papers.

Paper 1

Title:
{first_paper['title']}

Abstract:
{first_paper['abstract']}


Paper 2

Title:
{second_paper['title']}

Abstract:
{second_paper['abstract']}


Compare them based on:

1. Research Objective

2. Methodology

3. Key Contributions

4. Advantages

5. Limitations

6. Applications

Present the comparison in a clear table.
"""

        response = llm.invoke(comparison_prompt)
        return response.content

    return compare_papers


def build_all_tools(model, index, df, summarizer, kw_model, llm):
    """Build and return the full list of agent tools."""
    return [
        build_search_and_summarize_tool(model, index, df, summarizer),
        build_extract_keywords_tool(kw_model),
        build_compare_papers_tool(model, index, df, llm),
    ]
