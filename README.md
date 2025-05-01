# MediBuddy – Your AI-Powered Medical Assistant 🚀

MediBuddy is a cutting-edge conversational AI chatbot designed to provide reliable medical insights. Built with state-of-the-art language models and enriched with verified medical literature, MediBuddy aims to educate and assist users in navigating medical topics safely.

---

## 💡 What is MediBuddy?

MediBuddy is a **Streamlit-based chatbot** that enables users to ask medical questions and receive contextual answers. It is designed for:
- **Education**: Offering insights based on verified medical knowledge.
- **Preliminary Awareness**: Providing guidance before consulting healthcare professionals.

---

## 🔧 Tech Stack Overview

- **Frontend/UI**: Streamlit – Lightweight and interactive chatbot interface.
- **Data Source**: The Gale Encyclopedia of Medicine (PDFs parsed and chunked for NLP).
- **Document Loading**: Langchain’s `PyPDFLoader` and `DirectoryLoader`.
- **Text Chunking**: `RecursiveCharacterTextSplitter`.
- **Embeddings**: HuggingFace Embeddings using `sentence-transformers/all-MiniLM-L6-v2`.
- **Vector Database**: FAISS – Fast similarity-based search.
- **LLM Integration**: HuggingFace Endpoint with `mistralai/Mistral-7B-Instruct-v0.3`.
- **Prompt Engineering**: Custom `PromptTemplate` for medically-focused Q&A retrieval.
- **Environment Management**: `.env` files and `dotenv` for token security.

---

## 🧠 Features

- ✅ Natural-language medical Q&A.
- ✅ Retrieval Augmented Generation (RAG) pipeline for precise, context-based responses.
- ✅ Indexed and searchable medical knowledge base.
- ✅ Modular, scalable, and open for continuous improvement.
- ✅ Friendly UI with persistent chat history using `st.session_state`.

---

## 🎯 Why MediBuddy?

In an era of rampant health misinformation, MediBuddy empowers users to explore health topics safely and accurately. It is backed by structured, medically-reviewed knowledge to ensure trust and reliability.

---

## 📌 Next Steps

- 🔹 Adding multilingual support.
- 🔹 Incorporating voice input/output for accessibility.
- 🔹 Fine-tuning domain-specific prompts for enhanced accuracy.

---

## 💬 Get Involved!

We value your feedback, ideas, and collaboration opportunities! Let’s work together to build tools that genuinely help people navigate healthcare challenges. 💙

---

## 🏷️ Tags

#AI #LangChain #Streamlit #HuggingFace #MedicalAI #Chatbot #MediBuddy #NLP #LLM #FAISS #RAG #HealthcareInnovation #OpenSource #Python #MachineLearning #GaleEncyclopedia #Mistral7B
