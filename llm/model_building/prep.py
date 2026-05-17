
import os
import warnings
warnings.filterwarnings("ignore")

# Tokenizer
import tiktoken

# LangChain Components
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyMuPDFLoader
)
from langchain_huggingface import (
    HuggingFaceEmbeddings
)
from langchain_chroma import Chroma

# Hugging Face

from huggingface_hub import (
    hf_hub_download,
    HfApi
)

# Hugging Face Authentication
hf_token = os.getenv("HF_TOKEN")

if not hf_token:
    raise ValueError("HF_TOKEN is missing!")

api = HfApi(token=hf_token)

# Dataset Configuration
DATASET_PATH = "hf://datasets/Andrew2505/CKD-LLM/ckd.pdf"
OUTPUT_DIR = "ckd_rag_db"
REPO_ID = "Andrew2505/CKD-LLM"
EMBEDDING_MODEL_NAME = "thenlper/gte-large"
PDF_FILENAME = "ckd.pdf"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 2

# Create Output Directory
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Download PDF from Hugging Face Hub
print("Downloading PDF from Hugging Face Hub...")

pdf_path = hf_hub_download(
    repo_id=REPO_ID,
    repo_type="dataset",
    filename=PDF_FILENAME,
    token=hf_token
)

print(f"PDF downloaded to: {pdf_path}")

# Load PDF Document
print("Loading PDF document...")

pdf_loader = PyMuPDFLoader(pdf_path)
documents = pdf_loader.load()

# Text Chunking
print("Performing document chunking...")

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base",
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ".", " ", ""]
)

document_chunks = text_splitter.split_documents(documents)

print(f"Total chunks created: {len(document_chunks)}")


# Remove Duplicate Chunks
print("Removing duplicate chunks...")

unique_chunks = []
seen = set()

for chunk in document_chunks:
    if chunk.page_content not in seen:
        unique_chunks.append(chunk)
        seen.add(chunk.page_content)

document_chunks = unique_chunks

print(
    f"Unique chunks after deduplication: "
    f"{len(document_chunks)}"
)

# Embedding Model
print("Loading embedding model...")

embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    encode_kwargs={
        "normalize_embeddings": True
    }
)

print("Embedding model loaded successfully.")

# Create Chroma Vector Database
print("Creating Chroma vector database...")

vectorstore = Chroma.from_documents(
    documents=document_chunks,
    embedding=embedding_model,
    persist_directory=OUTPUT_DIR
)

print("DB COUNT:", vectorstore._collection.count())
print("Chroma vector database created successfully.")

# Load Persistent Vector Database
vectorstore = Chroma(
    persist_directory=OUTPUT_DIR,
    embedding_function=embedding_model
)

print("Persistent vector database loaded successfully.")

# Create Retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 2}
)

print("Retriever created successfully.")

# Test Retrieval
query = (
    "What are the symptoms of "
    "chronic kidney disease?"
)

print(f"\nTest Query: {query}")
retrieved_docs = retriever.invoke(query)
print("\nRetrieved Chunks:\n")

for idx, doc in enumerate(retrieved_docs):
    print(f"Chunk {idx+1}")
    print(doc.page_content)
    print("-" * 50)

# Upload Vector Database Files
print("\nUploading vector database files...")

files = []

for root, dirs, filenames in os.walk(OUTPUT_DIR):
    for filename in filenames:
        file_path = os.path.join(root, filename)
        files.append(file_path)

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
        repo_type="dataset",
    )

print("Upload completed successfully.")
