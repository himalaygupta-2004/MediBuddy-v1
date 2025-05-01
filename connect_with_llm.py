import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS

# Load .env file
load_dotenv()

# Hugging Face credentials and model ID
HF_TOKEN = os.getenv('HF_TOKEN')
if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found in environment. Please set it in your .env file.")

hugging_face_repo_id = "mistralai/Mistral-7B-Instruct-v0.3"

# 1. Load the Mistral model from Hugging Face
def load_llm(repo_id):
    return HuggingFaceEndpoint(
        repo_id=repo_id,
        task="text-generation",  # ✅ Add this line
        temperature=0.5,
        max_new_tokens=512,
        huggingfacehub_api_token=HF_TOKEN
    )

# 2. Set up the custom prompt template
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you don't know the answer, just say you don't know — don't try to make one up.
Only use the given context.

Context:
{context}

Question:
{question}

Start your answer directly — no small talk.
"""

def set_custom_prompt(template):
    return PromptTemplate(template=template, input_variables=["context", "question"])

# 3. Load the FAISS vector store
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# 4. Create the RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(hugging_face_repo_id),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

# 5. Invoke the chain
user_query = input("Write your query here: ")
response = qa_chain.invoke({"query": user_query})

print("\nAnswer:\n", response["result"])
print("\n Source Documents: ",response["source_documents"])
