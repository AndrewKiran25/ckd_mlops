
# CKD RAG PIPELINE
import os
import warnings
warnings.filterwarnings("ignore")

# LANGCHAIN IMPORTS
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_huggingface import (
    HuggingFaceEmbeddings
)

# HUGGING FACE IMPORTS
from huggingface_hub import (
    hf_hub_download
)

# LOCAL LLM IMPORT
from llama_cpp import Llama

# CONFIGURATION
VECTOR_DB_DIR = "ckd_db"

MODEL_REPO_ID = (
    "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
)

MODEL_FILE = (
    "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
)

Embedding_model_name = (
    "thenlper/gte-large"
)

TOP_K = 3
MAX_TOKENS = 512
TEMPERATURE = 0.2
CONTEXT_WINDOW = 4096

# LOAD EMBEDDING MODEL
print("=" * 60)
print("Loading embedding model...")
print("=" * 60)

embedding_model = HuggingFaceEmbeddings(
    model_name=Embedding_model_name,
    encode_kwargs={"normalize_embeddings": True}
)

print("Embedding model loaded successfully.")

# LOAD CHROMA VECTOR DATABASE
print("\n" + "=" * 60)
print("Loading Chroma vector database...")
print("=" * 60)

vectorstore = Chroma(
    persist_directory="ckd_db",
    embedding_function=embedding_model
)

print("Vector database loaded successfully.")

# CREATE RETRIEVER
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": TOP_K}
)

print("Retriever initialized successfully.")

# DOWNLOAD LLM MODEL
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


# PROMPT TEMPLATE
prompt_template = """
You are a helpful AI assistant specialized
in Chronic Kidney Disease (CKD).

Answer the question ONLY using the
provided context.

If the answer is not found in the
context, reply with:

"I don't know."

Context:
{context}

Question:
{question}

Answer:
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=[
        "context",
        "question"
    ]
)

# GENERATE RESPONSE FUNCTION
def generate_response(query):

    # RETRIEVE DOCUMENTS
    retrieved_docs = retriever.invoke(query)

    # COMBINE CONTEXT
    context = "\n\n".join(
        [
            doc.page_content
            for doc in retrieved_docs
        ]
    )
    # FORMAT PROMPT
    formatted_prompt = prompt.format(
        context=context,
        question=query
    )
    # GENERATE RESPONSE
    try:

        response = llm(
        formatted_prompt,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        stop=[
            "Question:",
            "Context:"
            ]
        )

        print(response)

        answer = (
            response["choices"][0]["text"]
            .strip()
        )

        return answer, retrieved_docs

    except Exception as e:
        print(f"LLM Error: {e}")
        return "Error generating response.", []

# TEST QUERY
if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("CKD RAG PIPELINE TEST")
    print("=" * 60)

    test_query = (
        "What are the symptoms of "
        "chronic kidney disease?"
    )

    print("\nTest Query:\n")
    print(test_query)

    try:

        answer, docs = generate_response(
            test_query
        )

        print("\nGenerated Answer:\n")
        print(answer)

        print("\nRetrieved Sources:\n")

        for idx, doc in enumerate(docs):

            print(f"Source Chunk {idx + 1}:\n")

            print(
                doc.page_content[:500]
            )

            print("\n" + "-" * 60)

        print("\nPipeline completed successfully.")

    except Exception as e:

        print(f"\nError: {e}")
