import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os
from dotenv import load_dotenv


# --- Load and prepare PDF documents ---
@st.cache_resource
def load_documents():
    loader = PyPDFLoader("UG-Porspectus-2024-25.pdf")
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(documents)

# --- Build Vectorstore ---
@st.cache_resource
def build_vectorstore(_docs):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(_docs, embedding=embeddings)
    return vectorstore.as_retriever()

load_dotenv()

# --- Create LLM and QA chain ---
def create_chain(retriever):
    from langchain.chains import ConversationalRetrievalChain
    from langchain.chat_models import ChatOpenAI

    # Load API key from .env
    openai_api_key = os.getenv("OPENROUTER_API_KEY")

    if openai_api_key is None:
        raise ValueError("OPENROUTER_API_KEY not found in .env file")

    llm = ChatOpenAI(
        model="deepseek/deepseek-chat-v3-0324:free",
        openai_api_base=os.environ["OPENAI_API_BASE"],
        openai_api_key=openai_api_key
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
    )

    return chain

def main():
    st.set_page_config(page_title="UNIVERSITY OF ENGINEERING & TECHNOLOGY", page_icon="🎓")
    st.title("UETM Chatbot")
    st.markdown("Ask me anything about UETM!")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        docs = load_documents()
        retriever = build_vectorstore(docs)
        st.session_state.qa_chain = create_chain(retriever)

    user_query = st.chat_input("Ask your question here...")

    if user_query:
        result = st.session_state.qa_chain({
            "question": user_query,
            "chat_history": st.session_state.chat_history
        })
        st.session_state.chat_history.append(("You", user_query))
        st.session_state.chat_history.append(("UETM Bot", result["answer"]))

    for role, message in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(message)

# --- Run the app ---
if __name__ == "__main__":
    main()