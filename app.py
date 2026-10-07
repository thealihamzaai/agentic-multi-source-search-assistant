import json
import os
from typing import Dict

import streamlit as st
from dotenv import load_dotenv
from langchain_community.tools import ArxivQueryRun, DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_groq import ChatGroq

load_dotenv()

st.set_page_config(page_title="Agentic Multi-Source Search Assistant", page_icon="🔎", layout="centered")

TOOLS = {
    "web": DuckDuckGoSearchRun(name="web_search"),
    "wikipedia": WikipediaQueryRun(
        api_wrapper=WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=2500)
    ),
    "arxiv": ArxivQueryRun(
        api_wrapper=ArxivAPIWrapper(top_k_results=3, doc_content_chars_max=3500)
    ),
}


def get_llm(api_key: str, model_name: str) -> ChatGroq:
    return ChatGroq(
        groq_api_key=api_key,
        model_name=model_name,
        temperature=0.2,
    )


def choose_tool(llm: ChatGroq, question: str) -> Dict[str, str]:
    prompt = f"""
You are a search-routing assistant. Choose the single best tool for the user's question.
Available tools:
- web: current events, recent information, companies, products, broad web research
- wikipedia: established background knowledge, people, places, historical topics
- arxiv: academic and technical research papers

Return ONLY valid JSON in this exact format:
{{"tool":"web|wikipedia|arxiv","query":"optimized search query"}}

User question: {question}
"""
    raw = llm.invoke(prompt).content.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw.replace("json\n", "", 1)
    try:
        data = json.loads(raw)
        if data.get("tool") not in TOOLS:
            raise ValueError
        query = str(data.get("query", "")).strip() or question
        return {"tool": data["tool"], "query": query}
    except Exception:
        return {"tool": "web", "query": question}


def run_tool(tool_name: str, query: str) -> str:
    try:
        return str(TOOLS[tool_name].invoke(query))
    except Exception as exc:
        return f"Tool error: {exc}"


def synthesize_answer(llm: ChatGroq, question: str, tool_name: str, evidence: str) -> str:
    prompt = f"""
Answer the user's question using the retrieved evidence below.
Be concise, clear, and factual. If the evidence is incomplete, say so.
Do not invent citations or facts. Mention which source type was used ({tool_name}).

Question:
{question}

Retrieved evidence:
{evidence}
"""
    return llm.invoke(prompt).content.strip()


st.title("🔎 Agentic Multi-Source Search Assistant")
st.caption("Routes questions to web search, Wikipedia, or arXiv, then synthesizes a concise answer.")

with st.sidebar:
    st.header("Settings")
    default_key = os.getenv("GROQ_API_KEY", "")
    api_key = st.text_input("Groq API key", value=default_key, type="password")
    model_choice = st.selectbox(
        "Groq model",
        ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"],
        index=0,
    )
    st.caption("Your API key is used only for the current app session.")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask me a question. I can route it to the web, Wikipedia, or arXiv.",
        }
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask a question")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if not api_key:
            answer = "Add a Groq API key in the sidebar to run the assistant."
            st.error(answer)
        else:
            try:
                with st.spinner("Searching and synthesizing..."):
                    llm = get_llm(api_key, model_choice)
                    route = choose_tool(llm, question)
                    evidence = run_tool(route["tool"], route["query"])
                    answer = synthesize_answer(llm, question, route["tool"], evidence)
                st.markdown(answer)
                with st.expander("Search details"):
                    st.write(f"Source type: {route['tool']}")
                    st.write(f"Search query: {route['query']}")
            except Exception as exc:
                answer = f"The request could not be completed: {exc}"
                st.error(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
