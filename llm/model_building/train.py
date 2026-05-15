import os
import warnings

warnings.filterwarnings("ignore")

# LangChain Components
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate

# Recommended Embeddings Import
from langchain_huggingface import HuggingFaceEmbeddings

# Hugging Face
from huggingface_hub import hf_hub_download, HfApi

# Local LLM
from llama_cpp import Llama

# =========================================================
# CONFIGURATION
# =========================================================

VECTOR_DB_DIR = "ckd_rag_db"

MODEL_REPO_ID = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
MODEL_FILE = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"

EMBEDDING_MODEL_NAME = "thenlper/gte-large"

TOP_K = 2
MAX_TOKENS = 512
TEMPERATURE = 0.2
CONTEXT_WINDOW = 4096

# Hugging Face Upload Settings
HF_TOKEN = os.getenv("HF_TOKEN")

REPO_ID = "your-username/your-model-repo"

OUTPUT_DIR = "output"

# =========================================================
# LOAD EMBEDDING MODEL
# =========================================================

print("Loading embedding model...")

embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME
)

print("Embedding model loaded successfully.")

# =========================================================
# LOAD CHROMA VECTOR DATABASE
# =========================================================

print("Loading Chroma vector database...")

vectorstore = Chroma(
    persist_directory=VECTOR_DB_DIR,
    embedding_function=embedding_model
)

print("Vector database loaded successfully.")

# =========================================================
# CREATE RETRIEVER
# =========================================================

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": TOP_K}
)

print("Retriever initialized successfully.")

# =========================================================
# DOWNLOAD GGUF MODEL
# =========================================================

print("Downloading GGUF model from Hugging Face...")

model_path = hf_hub_download(
    repo_id=MODEL_REPO_ID,
    filename=MODEL_FILE
)

print(f"Model downloaded successfully: {model_path}")

# =========================================================
# LOAD LOCAL LLM
# =========================================================

print("Loading local LLM...")

llm = Llama(
    model_path=model_path,
    n_ctx=CONTEXT_WINDOW,
    n_threads=os.cpu_count(),
    verbose=False
)

print("Local LLM loaded successfully.")

# =========================================================
# PROMPT TEMPLATE
# =========================================================

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

# =========================================================
# RAG RESPONSE FUNCTION
# =========================================================

def generate_response(query):

    # Retrieve Documents
    retrieved_docs = retriever.invoke(query)

    # Combine Context
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

# =========================================================
# OPTIONAL: UPLOAD FILES TO HUGGING FACE
# =========================================================

def upload_to_huggingface():

    if not HF_TOKEN:
        raise ValueError(
            "HF_TOKEN environment variable is missing!"
        )

    if not os.path.exists(OUTPUT_DIR):
        print(f"Output directory '{OUTPUT_DIR}' not found.")
        return

    api = HfApi(token=HF_TOKEN)

    files = []

    # Collect Files
    for root, dirs, filenames in os.walk(OUTPUT_DIR):

        for filename in filenames:

            full_path = os.path.join(root, filename)

            files.append(full_path)

    if len(files) == 0:
        print("No files found for upload.")
        return

    # Upload Files
    for file_path in files:

        relative_path = os.path.relpath(
            file_path,
            OUTPUT_DIR
        )

        print(f"Uploading: {relative_path}")

        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=relative_path,
            repo_id=REPO_ID,
            repo_type="model"
        )

    print("Upload completed successfully.")

# =========================================================
# INTERACTIVE CHAT LOOP
# =========================================================

print("\nCKD RAG Chatbot is Ready!")
print("Type 'exit' to quit.")
print("Type 'upload' to upload output files.\n")

while True:

    user_query = input("User: ")

    # Exit
    if user_query.lower() == "exit":

        print("Exiting chatbot...")
        break

    # Upload Command
    elif user_query.lower() == "upload":

        try:
            upload_to_huggingface()

        except Exception as e:
            print(f"Upload Error: {e}")

        continue

    # Generate Response
    try:

        answer, docs = generate_response(user_query)

        print("\nAssistant:")
        print(answer)

        print("\nRetrieved Sources:\n")

        for idx, doc in enumerate(docs):

            print(f"Source Chunk {idx + 1}:")
            print(doc.page_content[:500])
            print("-" * 50)

        print("\n")

    except Exception as e:

        print(f"Error: {e}")
