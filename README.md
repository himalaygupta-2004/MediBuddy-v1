🚀 Excited to share my latest project: MediBuddy – Your AI-Powered Medical Assistant! 🧠💬

Over the past few weeks, I’ve been building MediBuddy, a conversational AI chatbot designed to assist users with reliable medical insights – powered by state-of-the-art language models and enriched with trusted data from The Gale Encyclopedia of Medicine 📘.

💡 What is MediBuddy?
MediBuddy is a streamlit-based chatbot that allows users to ask medical questions and receive contextual answers based on verified medical literature. It's designed for education, preliminary awareness, and informational support — not as a replacement for doctors, but as a bridge to understanding health topics.

🔧 Tech Stack Overview
Frontend/UI: Streamlit – lightweight and interactive chatbot interface

Data Source: The Gale Encyclopedia of Medicine (PDFs parsed and chunked for NLP)

Document Loading: Langchain’s PyPDFLoader and DirectoryLoader

Text Chunking: RecursiveCharacterTextSplitter

Embeddings: HuggingFaceEmbeddings using "sentence-transformers/all-MiniLM-L6-v2"

Vector Database: FAISS for fast, similarity-based search

LLM Integration: HuggingFaceEndpoint with "mistralai/Mistral-7B-Instruct-v0.3"

Prompt Engineering: Custom PromptTemplate for medically-focused Q&A retrieval

Environment Management: .env and dotenv for token security

🧠 Features
✅ Natural-language medical Q&A
✅ RAG (Retrieval Augmented Generation) pipeline for precise context-based responses
✅ Indexed and searchable medical knowledge base
✅ Modular, scalable, and open for continuous improvement
✅ Friendly UI with persistent chat history using st.session_state

🎯 Why MediBuddy?
In an age where health misinformation is rampant, I wanted to build a tool that empowers people to explore health topics safely and accurately, backed by structured, medically-reviewed knowledge.

📌 Next Steps 🔹 Adding multilingual support
🔹 Voice input/output for accessibility
🔹 More fine-tuning on domain-specific prompts

💬 I'd love your feedback, ideas, or even collaboration opportunities! Let’s build tools that genuinely help people! 💙

#AI #LangChain #Streamlit #HuggingFace #MedicalAI #Chatbot #MediBuddy #NLP #LLM #FAISS #RAG #HealthcareInnovation #OpenSource #Python #MachineLearning #GaleEncyclopedia #Mistral7B
