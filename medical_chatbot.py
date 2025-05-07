import streamlit as st
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceEndpoint
import os
import re
from dotenv import load_dotenv

load_dotenv()

DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt(custom_prompt_template):
    return PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])

def load_llm(repo_id, hf_token):
    return HuggingFaceEndpoint(
        repo_id=repo_id,
        task="text-generation",
        temperature=0.5,
        max_new_tokens=512,
        huggingfacehub_api_token=hf_token
    )

def format_response(result, sources):
    formatted_result = "### 🩺 **MediBot:**\n\n"

    if re.search(r'^\s*1[\.\)]', result.strip()):
        parts = re.split(r'(?=\s*\d{1,2}[\.\)])', result.strip())
        for part in parts:
            if part.strip():
                formatted_result += f"{part.strip()}\n\n"
    else:
        sentences = re.split(r'(?<=[.!?]) +|\n+', result.strip())
        for idx, sentence in enumerate(sentences, 1):
            if sentence.strip():
                formatted_result += f"{idx}. {sentence.strip()}\n\n"

    formatted_sources = "### 📑 **Source Preview:**\n\n"
    for i, doc in enumerate(sources, 1):
        preview = doc.page_content[:200].replace("\n", " ").strip() + "..."
        source = doc.metadata.get("source", "Unknown Source")
        formatted_sources += f"{i}. **Source** ({source}):\n> {preview}\n\n"

    return formatted_result + formatted_sources

def main():
    st.title("Ask MediBot 🧠")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt = st.chat_input("How can I assist you with your health today?")

    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        CUSTOM_PROMPT_TEMPLATE = """
        Use the pieces of information provided in the context to answer the user's question.
        If you don't know the answer, just say you don't know — don't try to make one up.
        Only use the given context.

        Context:
        {context}

        Question:
        {question}

        Start your answer directly — no small talk.
        """

        try:
            hf_token = os.getenv("HF_TOKEN")
            hugging_face_repo_id = "mistralai/Mistral-7B-Instruct-v0.3"

            db = get_vectorstore()
            llm = load_llm(repo_id=hugging_face_repo_id, hf_token=hf_token)
            prompt_template = set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)

            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=db.as_retriever(search_kwargs={"k": 3}),
                return_source_documents=True,
                chain_type_kwargs={"prompt": prompt_template}
            )

            response = qa_chain.invoke({'query': prompt})
            result = response["result"]
            sources = response["source_documents"]

            full_response = format_response(result, sources)

            st.chat_message("assistant").markdown(full_response)
            st.session_state.messages.append({'role': 'assistant', 'content': full_response})

        except Exception as e:
            st.error(f"⚠️ Error encountered: {e}")

if __name__ == "__main__":
    main()
