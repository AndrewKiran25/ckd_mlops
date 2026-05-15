# CKD RAG CHATBOT - FINAL CLEAN VERSION
import os
import warnings
warnings.filterwarnings("ignore")

# LANGCHAIN IMPORTS
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# HUGGING FACE IMPORTS
from huggingface_hub import hf_hub_download

# LOCAL LLM
from llama_cpp import Llama

# CONFIGURATION
VECTOR_DB_DIR = "ckd_rag_db"
MODEL_REPO_ID = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
MODEL_FILE = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
EMBEDDING_MODEL_NAME = "thenlper/gte-large"
TOP_K = 3
MAX_TOKENS = 512
TEMPERATURE = 0.2
CONTEXT_WINDOW = 4096

# LOAD EMBEDDING MODEL
print("=" * 60)
print("Loading embedding model...")
print("=" * 60)

embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME
)

print("Embedding model loaded successfully.")

# LOAD CHROMA VECTOR DATABASE
print("\n" + "=" * 60)
print("Loading Chroma vector database...")
print("=" * 60)

vectorstore = Chroma(
    persist_directory=VECTOR_DB_DIR,
    embedding_function=embedding_model
)

print("Vector database loaded successfully.")

# CREATE RETRIEVER
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": TOP_K}
)

print("Retriever initialized successfully.")

# DOWNLOAD GGUF MODEL
print("\n" + "=" * 60)
print("Downloading GGUF model...")
print("=" * 60)

model_path = hf_hub_download(
    repo_id=MODEL_REPO_ID,
    filename=MODEL_FILE
)

print(f"Model downloaded successfully:\n{model_path}")

# LOAD LOCAL LLM
print("\n" + "=" * 60)
print("Loading local LLM...")
print("=" * 60)

llm = Llama(
    model_path=model_path,
    n_ctx=CONTEXT_WINDOW,
    n_threads=os.cpu_count(),
    verbose=False
)

print("Local LLM loaded successfully.")

# RESPONSE GENERATION FUNCTION
def generate_response(user_query):
    retrieved_docs = retriever.invoke(user_query)

    context = "\n\n".join(
        [
            doc.page_content
            for doc in retrieved_docs
        ]
    )



prompt = f"""
You are a helpful medical assistant.

Answer the question ONLY from the provided context.

If the answer is not available in the context,
reply with:

"I don't know."

Context:
{context}

Question:
{user_query}

Answer:
"""

    # Generate LLM Response


    response = llm(
        prompt,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        stop=["Question:", "Context:"]
    )

    answer = (
        response["choices"][0]["text"]
        .strip()
    )

    return answer, retrieved_docs


# TEST QUERY
print("\n" + "=" * 60)
print("Running test query...")
print("=" * 60)

test_query = (
    "What are the symptoms of chronic kidney disease?"
)

answer, docs = generate_response(test_query)

print(f"\nQuestion:\n{test_query}")

print("\nAnswer:\n")
print(answer)

print("\nRetrieved Sources:\n")

for idx, doc in enumerate(docs):

    print(f"Source Chunk {idx + 1}:\n")

    print(doc.page_content[:500])

    print("\n" + "-" * 60)


# INTERACTIVE CHAT LOOP
print("\n" + "=" * 60)
print("CKD RAG CHATBOT READY")
print("=" * 60)

print("\nType 'exit' to quit.\n")

while True:

    user_query = input("User: ")

    # ---------------------------------------------
    # Exit Condition
    # ---------------------------------------------

    if user_query.lower() == "exit":

        print("\nExiting chatbot...")
        break

    # ---------------------------------------------
    # Generate Response
    # ---------------------------------------------

    try:

        answer, docs = generate_response(
            user_query
        )

        print("\nAssistant:\n")

        print(answer)

        print("\nRetrieved Sources:\n")

        for idx, doc in enumerate(docs):

            print(f"Source Chunk {idx + 1}:\n")

            print(
                doc.page_content[:500]
            )

            print("\n" + "-" * 60)

    except Exception as e:

        print(f"\nError: {e}")
# OPTIONAL: UPLOAD FILES TO HUGGING FACE
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

