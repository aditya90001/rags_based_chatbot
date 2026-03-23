import streamlit as st
import requests
import time

# 🔥 Config
st.set_page_config(page_title="AI Chatbot", layout="wide")
st.title("Multi Agent Chatbot (RAG + Web Search)")

API_URL = "http://127.0.0.1:8000/chat"

# 🔥 Sidebar (Settings)
st.sidebar.header("⚙️ Settings")

system_prompt = st.sidebar.text_area(
    "System Prompt",
    "You are a helpful AI assistant"
)

provider = st.sidebar.selectbox("Provider", ["Groq", "hugging_face"])

if provider == "Groq":
    model = st.sidebar.selectbox(
        "Model",
        ["llama-3.3-70b-versatile"]
    )
else:
    model = st.sidebar.selectbox(
        "Model",
        ["deepseek-ai/DeepSeek-R1"]
    )

allow_search = st.sidebar.checkbox("Enable Web Search")
use_rag = st.sidebar.checkbox("Enable RAG (PDF + TXT Retrieval)", value=True)

# 🔥 Clear Chat Button
if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state.chat_history = []

# 🔥 Chat Memory
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 🔥 Display previous messages
for chat in st.session_state.chat_history:
    with st.chat_message(chat["role"]):
        st.markdown(chat["content"])

# 🔥 Chat input
user_input = st.chat_input("Type your message...")

if user_input:
    # Show user message immediately
    st.chat_message("user").markdown(user_input)
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # 🔥 Prepare request payload
    payload = {
        "model_name": model,
        "model_provider": provider,
        "system_prompt": system_prompt,
        "messages": [user_input],
        "allow_search": allow_search,
        "use_rag": use_rag  # Pass RAG toggle to backend
    }

    # 🔥 Assistant response
    with st.chat_message("assistant"):
        with st.spinner("🤖 Thinking..."):
            try:
                response = requests.post(API_URL, json=payload, timeout=60)
                data = response.json()

                if "error" in data:
                    reply = data["error"]
                    main, sources = reply, None
                else:
                    reply = data["response"]
                    # 🔥 Split sources if present
                    if "Sources:" in reply:
                        main, sources = reply.split("Sources:", 1)
                    else:
                        main, sources = reply, None

                # 🔥 Typing effect for main response
                placeholder = st.empty()
                typed_text = ""
                for char in main:
                    typed_text += char
                    placeholder.markdown(typed_text)
                    time.sleep(0.005)

                # 🔥 Show sources in an expander
                if sources:
                    with st.expander("📚 Sources"):
                        st.markdown(sources.strip())

            except requests.exceptions.Timeout:
                reply = "Connection timed out. Please try again."
                st.error(reply)
            except Exception as e:
                reply = f"Connection error: {str(e)}"
                st.error(reply)

    # 🔥 Save assistant response in chat history
    st.session_state.chat_history.append({"role": "assistant", "content": reply})