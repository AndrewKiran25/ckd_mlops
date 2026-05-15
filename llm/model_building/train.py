
import os
import warnings

warnings.filterwarnings("ignore")

# LangChain Components
from langchain_community.embeddings import (
    SentenceTransformerEmbeddings
)
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Hugging Face
from huggingface_hub import hf_hub_download

# Local LLM
from llama_cpp import Llama

# Configuration

VECTOR_DB_DIR = "ckd_rag_db"
MODEL_REPO_ID = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
MODEL_FILE = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
EMBEDDING_MODEL_NAME = "thenlper/gte-large"
TOP_K = 2
MAX_TOKENS = 512
TEMPERATURE = 0.2
CONTEXT_WINDOW = 4096

# Load Embedding Model

print("Loading embedding model...")

embedding_model = SentenceTransformerEmbeddings(
    model_name=EMBEDDING_MODEL_NAME
)

print("Embedding model loaded successfully.")

# Load Chroma Vector Database

print("Loading Chroma vector database...")

vectorstore = Chroma(
    persist_directory=VECTOR_DB_DIR,
    embedding_function=embedding_model
)

print("Vector database loaded successfully.")

# Create Retriever

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": TOP_K}
)

print("Retriever initialized successfully.")

# Download GGUF Model from Hugging Face

print("Downloading GGUF model from Hugging Face...")

model_path = hf_hub_download(
    repo_id=MODEL_REPO_ID,
    filename=MODEL_FILE
)

print(f"Model downloaded successfully: {model_path}")

# Load Local LLM

print("Loading local LLM...")

llm = Llama(
    model_path=model_path,
    n_ctx=CONTEXT_WINDOW,
    n_threads=os.cpu_count(),
    verbose=False
)

print("Local LLM loaded successfully.")

# Prompt Template

prompt_template = """
You are a helpful AI assistant specialized in
Chronic Kidney Disease (CKD).

Use the provided context to answer the question.

If the answer is not available in the context,
say:
"I could not find the answer in the provided documents."

Context:
{context}

Question:
{question}

Answer:
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# RAG Inference Function

def generate_response(query):

    # Retrieve Documents
    retrieved_docs = retriever.invoke(query)

    context = "\n\n".join(
        [doc.page_content for doc in retrieved_docs]
    )

    # Format Prompt
    formatted_prompt = prompt.format(
        context=context,
        question=query
    )

    # Generate Response
    response = llm(
        formatted_prompt,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        stop=["Question:", "Context:"]
    )

    answer = response["choices"][0]["text"].strip()

    return answer, retrieved_docs


# Interactive Chat Loop

print("\nCKD RAG Chatbot is Ready!")
print("Type 'exit' to quit.\n")

while True:

    user_query = input("User: ")

    if user_query.lower() == "exit":
        print("Exiting chatbot...")
        break

    answer, docs = generate_response(user_query)

    print("\nAssistant:")
    print(answer)

    print("\nRetrieved Sources:\n")

    for idx, doc in enumerate(docs):
        print(f"Source Chunk {idx+1}:")
        print(doc.page_content[:500])
        print("-" * 50)

    print("\n")

if not hf_token:
    raise ValueError("HF_TOKEN is missing!")

api = HfApi(token=hf_token)

# Upload Files to Hugging Face
for file_path in files:

    relative_path = os.path.relpath(
        file_path,
        OUTPUT_DIR
    )

    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=relative_path,
        repo_id=REPO_ID,
        repo_type="model",
    )

print("Upload completed successfully.")
