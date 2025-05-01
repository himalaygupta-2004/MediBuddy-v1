# MediBuddy – Your AI-Powered Medical Assistant 🚀

MediBuddy is a cutting-edge conversational AI chatbot designed to provide reliable medical insights. Built with state-of-the-art language models and enriched with verified medical literature, MediBuddy empowers users to navigate healthcare topics safely and accurately.

---

## 📖 Table of Contents
- [What is MediBuddy?](#-what-is-medibuddy)
- [Tech Stack Overview](#-tech-stack-overview)
- [Features](#-features)
- [Why MediBuddy?](#-why-medibuddy)
- [Installation and Setup](#-installation-and-setup)
- [Next Steps](#-next-steps)
- [Get Involved](#-get-involved)
- [Demo](#-demo)
- [FAQ](#-faq)
- [License](#-license)
- [Tags](#-tags)

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

## 🚀 Installation and Setup

Follow these steps to set up MediBuddy locally:

1. Clone the repository:
   ```bash
   git clone https://github.com/himalaygupta-2004/MediBuddy-v1.git

Navigate to the project directory:

bash
cd MediBuddy-v1
Install dependencies:

bash
pip install -r requirements.txt
Run the application:

bash
streamlit run app.py

📌 Next Steps
🔹 Adding multilingual support.
🔹 Incorporating voice input/output for accessibility.
🔹 Fine-tuning domain-specific prompts for enhanced accuracy.
💬 Get Involved!
We value your feedback, ideas, and collaboration opportunities! Let’s work together to build tools that genuinely help people navigate healthcare challenges. 💙


❓ FAQ
What data sources does MediBuddy use? MediBuddy relies on verified medical literature, like the Gale Encyclopedia of Medicine.

Is this chatbot a substitute for medical advice? No, MediBuddy is for educational purposes and should not replace professional medical advice.

📜 License
This project is licensed under the MIT License. See the LICENSE file for more details.

🏷️ Tags
#AI #LangChain #Streamlit #HuggingFace #MedicalAI #Chatbot #MediBuddy #NLP #LLM #FAISS #RAG #HealthcareInnovation #OpenSource #Python #MachineLearning #GaleEncyclopedia #Mistral7B


