# campus-assistant

Just launched: UETM assistant – a conversational guide to University of Engineering & Technology Mardan admissions.

This PDF‑powered chatbot:

Parses the official UETM Prospectus 2024–25 to answer questions about programs, eligibility, campuses and application procedures.

Leverages LangChain’s in‑memory buffer for context retention.

Extracts data from PDF with PyPDFLoader.

Performs semantic search using Hugging Face embeddings and a FAISS vector store.

Delivers real‑time Q&A via OpenRouter’s LLM API.

Presents an interactive web interface built in Streamlit.

Secures API keys and configuration with python‑dotenv
