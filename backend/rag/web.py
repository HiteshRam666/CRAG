from langchain_tavily import TavilySearch
from langchain_core.documents import Document
from config import TAVILY_API_KEY
from pydantic import BaseModel 
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

tavily = TavilySearch(max_results=5, api_key = TAVILY_API_KEY)

# Query rewrite for web search
class WebQuery(BaseModel):
    query: str 

rewrite_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Rewrite the user question into a web search query composed of keywords.\n"
            "Rules:\n"
            "- Keep it short (6–14 words).\n"
            "- If the question implies recency (e.g., recent/latest/last week/last month), add a constraint like (last 30 days).\n"
            "- Do NOT answer the question.\n"
            "- Return JSON with a single key: query",
        ),
        ("human", "Question: {question}"),
    ]
)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# def rewrite_query(state):
#     return {"web_query": state["question"]}

rewrite_chain = rewrite_prompt | llm.with_structured_output(WebQuery)  

def rewrite_query(state):
    out = rewrite_chain.invoke({"question": state["question"]})
    return {"web_query": out.query} 

# def web_search_node(state):
#     q = state.get("web_query") or state["question"]

#     results = tavily.invoke(q)

#     print("RAW TAVILY OUTPUT:", results)

#     web_docs = []

#     # Case 1: dict with 'results'
#     if isinstance(results, dict) and "results" in results:
#         results = results["results"]

#     # Case 2: string
#     if isinstance(results, str):
#         return {
#             "web_docs": [
#                 Document(page_content=results, metadata={"source": "tavily"})
#             ]
#         }

#     # Case 3: list of dicts
#     if isinstance(results, list):
#         for r in results:
#             if isinstance(r, dict):
#                 title = r.get("title", "")
#                 url = r.get("url", "")
#                 content = r.get("content", "") or r.get("snippet", "")

#                 text = f"TITLE: {title}\nURL: {url}\nCONTENT:\n{content}"

#                 web_docs.append(
#                     Document(
#                         page_content=text,
#                         metadata={"url": url, "title": title}
#                     )
#                 )

#     return {"web_docs": web_docs}

def web_search_node(state):
    q = state.get("web_query") or state["question"]

    results = tavily.invoke(q)

    print("RAW TAVILY OUTPUT:", results)

    web_docs = []

    # Case 1: dict with 'results'
    if isinstance(results, dict) and "results" in results:
        results = results["results"]

    # Case 2: string
    if isinstance(results, str):
        return {
            "web_docs": [
                Document(
                    page_content=results, 
                    metadata={
                        "source": "tavily",
                        "url": "https://tavily.com/search"
                    }
                )
            ]
        }

    # Case 3: list of dicts - THIS IS THE IMPORTANT PART FOR URLS
    if isinstance(results, list):
        for r in results:
            if isinstance(r, dict):
                title = r.get("title", "")
                url = r.get("url", "")
                content = r.get("content", "") or r.get("snippet", "")

                text = f"TITLE: {title}\nURL: {url}\nCONTENT:\n{content}"

                web_docs.append(
                    Document(
                        page_content=text,
                        metadata={
                            "url": url, 
                            "title": title,
                            "source_type": "web"
                        }
                    )
                )

    return {"web_docs": web_docs}

