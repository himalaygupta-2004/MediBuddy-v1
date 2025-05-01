import streamlit as st
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
import os
from dotenv import load_dotenv

# Load environment variables
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

def main():
    st.title("Ask MediBot 🧠")

    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Display previous messages
    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    # User input
    prompt = st.chat_input("Pass your prompt here:")

    if prompt:
        # Show user message
        st.chat_message('user').markdown(prompt)
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
            hf_token = os.getenv('HF_TOKEN')
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
            # sources = "\n\n**Sources:**\n" + "\n".join([doc.metadata.get("source", "Unknown") for doc in response["source_documents"]])
            sources = response["source_documents"]

            full_response = result +"\n Source Docs: \n" + str(sources)

            st.chat_message('assistant').markdown(full_response)
            st.session_state.messages.append({'role': 'assistant', 'content': full_response})

        except Exception as e:
            st.error(f"Error encountered: {e}")

if __name__ == "__main__":
    main()
