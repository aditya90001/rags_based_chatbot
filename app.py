from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END, START
from typing import TypedDict, List, Dict, Any
from langchain_core.messages import BaseMessage
from dotenv import load_dotenv
import os
load_dotenv()
import streamlit as st 
import os 
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY") or st.secrets.get("TAVILY_API_KEY")
# 🤖 LLM
@st.cache_resource
def get_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=GROQ_API_KEY,
        streaming=True
    )

llm = get_llm()

# 🌐 Search Tool
tavily = TavilySearch(api_key=TAVILY_API_KEY, max_results=2) if TAVILY_API_KEY else None

# 📄 Load docs
@st.cache_data
def load_docs():
    docs = []
    if os.path.exists("docs"):
        for file in os.listdir("docs"):
            if file.endswith(".txt"):
                with open(os.path.join("docs", file), "r", encoding="utf-8") as f:
                    docs.append(f.read())
    return docs

# 🔍 Vector DB
@st.cache_resource
def create_vectorstore():
    docs = load_docs()
    if not docs:
        return None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50
    )

    chunks = splitter.create_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    return FAISS.from_documents(chunks, embeddings)

vectorstore = create_vectorstore()

# 🧠 ROUTER NODE
def router(state):
    query = state["query"]

    prompt = f"""
Decide best tool:
1. RAG
2. SEARCH
3. DIRECT

Query: {query}

Answer only one word: RAG or SEARCH or DIRECT
"""

    decision = llm.invoke([HumanMessage(content=prompt)]).content.upper()

    next_step = "direct"

    if "RAG" in decision:
        next_step = "rag"
    elif "SEARCH" in decision:
        next_step = "search"

    return {**state, "next": next_step}

# 📄 RAG NODE
def rag_node(state):
    query = state["query"]

    if vectorstore is None:
        return {**state, "context": ""}

    docs = vectorstore.similarity_search(query, k=2)
    context = "\n".join([d.page_content[:300] for d in docs])

    return {**state, "context": context}

# 🌐 SEARCH NODE
def search_node(state):
    query = state["query"]

    if tavily is None:
        return {**state, "context": ""}

    try:
        result = tavily.invoke(query)
        return {**state, "context": str(result)}
    except:
        return {**state, "context": ""}

# 🤖 LLM NODE
def llm_node(state):
    messages = state["messages"]
    context = state.get("context", "")

    if context:
        messages.append(HumanMessage(content=f"Context:\n{context}"))

    return {**state, "messages": messages}

# 🔗 GRAPH BUILD
builder = StateGraph(dict)

builder.add_node("router", router)
builder.add_node("rag", rag_node)
builder.add_node("search", search_node)
builder.add_node("llm", llm_node)

# START → router
builder.add_edge(START, "router")

# Conditional routing
builder.add_conditional_edges(
    "router",
    lambda state: state["next"],
    {
        "rag": "rag",
        "search": "search",
        "direct": "llm"
    }
)

# Flow
builder.add_edge("rag", "llm")
builder.add_edge("search", "llm")
builder.add_edge("llm", END)

graph = builder.compile()

# 🎨 UI
st.set_page_config(page_title="AI Agent", layout="wide")
st.title("🚀 AI Agent (RAG + Search + LangGraph)")

# 🧠 MEMORY
if "history" not in st.session_state:
    st.session_state.history = []

# 💬 SHOW CHAT
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 🧑 INPUT
query = st.chat_input("Ask anything...")

if query:
    st.chat_message("user").markdown(query)

    state = {
        "query": query,
        "messages": [
            SystemMessage(content="You are a helpful AI assistant."),
            *[
                HumanMessage(content=m["content"])
                if m["role"] == "user"
                else SystemMessage(content=m["content"])
                for m in st.session_state.history[-6:]
            ]
        ]
    }

    result = graph.invoke(state)
    final_messages = result["messages"]

    # 🔥 STREAMING RESPONSE
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_text = ""

        stream = llm.stream(final_messages)

        for chunk in stream:
            if chunk.content:
                full_text += chunk.content
                placeholder.markdown(full_text)

    # 💾 SAVE MEMORY
    st.session_state.history.append({"role": "user", "content": query})
    st.session_state.history.append({"role": "assistant", "content": full_text})

    # limit memory
    st.session_state.history = st.session_state.history[-10:]
