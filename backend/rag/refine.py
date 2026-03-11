import re
from typing import List 
from pydantic import BaseModel 
from langchain_openai import ChatOpenAI 
from langchain_core.prompts import ChatPromptTemplate

def decompose_to_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if len(s.strip()) > 20]

# FILTER (LLM Judge)
class KeepOrDrop(BaseModel):
    keep: bool

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


filter_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict relevance filter.\n"
            "Return keep=true only if the sentence directly helps answer the question.\n"
            "Use ONLY the sentence. Output JSON only.",
        ),
        ("human", "Question: {question}\n\nSentence:\n{sentence}"),
    ]
)

filter_chain = filter_prompt | llm.with_structured_output(KeepOrDrop)

# Knowledge refinement
# (CORRECT => internal only)
# (INCORRECT => web only)
# (AMBIGUOUS => internal + web)

def refine(state): 
    q = state["question"] 

    if state.get("verdict") == "CORRECT":
        docs_to_use = state["good_docs"] 
    elif state.get("verdict") == "INCORRECT":
        docs_to_use = state["web_docs"] 
    else: # Ambiguous
        docs_to_use = state["good_docs"] + state["web_docs"]

    context = "\n\n".join(d.page_content for d in docs_to_use).strip() 

    strips = decompose_to_sentences(context) 

    kept: List[str] = [] 
    for s in strips:
        if filter_chain.invoke({"question": q, "sentence": s}).keep:
            kept.append(s) 

    refined_context = "\n".join(kept).strip() 

    return {
        "strips": strips,
        "kept_strips": kept,
        "refined_context": refined_context,
    }