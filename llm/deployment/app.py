
# Import Libraries
import streamlit as st
import warnings
import time

warnings.filterwarnings("ignore")

# LangChain
from langchain_chroma import Chroma

# Transformers
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_huggingface import HuggingFaceEmbeddings

# Page Configuration
st.set_page_config(
    page_title="CKD RAG Chatbot",
    page_icon="🩺",
    layout="wide"
)


# App Title
st.title("🩺 Chronic Kidney Disease RAG Chatbot")
st.markdown(
    """
    Ask questions related to Chronic Kidney Disease (CKD).
    """
)

# Load Embedding Model
@st.cache_resource
def load_embedding_model():

    embedding_model = HuggingFaceEmbeddings(
        model_name="thenlper/gte-large",
        encode_kwargs={
            "normalize_embeddings": True
        }
    )

    return embedding_model

# Load Vector Database
@st.cache_resource
def load_vectorstore():

    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id="Andrew2505/CKD-LLM",
        repo_type="dataset",
        local_dir="ckd_rag_db"
    )

    embedding_model = load_embedding_model()

    vectorstore = Chroma(
        persist_directory="ckd_rag_db",
        embedding_function=embedding_model
    )

    #DEBUG (PUT HERE)
    query = "chronic kidney disease"

    docs = vectorstore.similarity_search(query, k=3)

    st.write("TOP MATCHES:")

    for i, d in enumerate(docs):
        st.write(f"Chunk {i+1}")
        st.write(d.page_content[:500])

    print("DB COUNT:", vectorstore._collection.count())

    return vectorstore

# Load Retriever
@st.cache_resource
def load_retriever():

    vectorstore = load_vectorstore()

    retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

    return retriever

# Load LLM
@st.cache_resource
def load_llm():

    pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_new_tokens=256,
        temperature=0
    )

    llm = HuggingFacePipeline(
        pipeline=pipe
    )

    return llm

# Prompt Templates
qna_system_message = """
You are an assistant whose work is to review the report and provide the appropriate answers from the context.

User input will have the context required by you to answer user questions.

This context will begin with the token: ###Context.

The context contains references to specific portions of a document relevant to the user query.

User questions will begin with the token: ###Question.

Please answer only using the context provided in the input.

Do not mention anything about the context in your final answer.

If the answer is not found in the context, respond "I don't know".
"""


qna_user_message_template = """
###Context
Here are some documents that are relevant to the question mentioned below.

{context}

###Question
{question}
"""


# Generate RAG Response
def generate_rag_response(
    query,
    retriever,
    llm
):

    # Retrieve Relevant Chunks
    relevant_document_chunks = (
        retriever.invoke(
            query
        )
    )

    print("\n" + "=" * 60)
    print("RETRIEVED DOCUMENTS")
    print("=" * 60)

    for idx, doc in enumerate(relevant_document_chunks):

        print(f"\nChunk {idx+1}:\n")

        print(doc.page_content[:1000])

        print("\n" + "-" * 50)

    if len(relevant_document_chunks) == 0:
        return "No relevant documents found."

    # Extract Chunk Content
    context_list = [
        doc.page_content
        for doc in relevant_document_chunks
    ]


    # Merge Context
    context_for_query = "\n".join(
        context_list
    )

    # Build User Prompt
    user_message = (
        qna_user_message_template
        .replace(
            "{context}",
            context_for_query
        )
        .replace(
            "{question}",
            query
        )
    )

    # Final Prompt
    prompt = (
        qna_system_message
        + "\n"
        + user_message
    )

    # Generate Response
    try:

        response = llm.invoke(prompt)

        response_text = str(response).strip()

    except Exception as e:

        response_text = (
            f"Error occurred: {e}"
        )

    return response_text

# Load Models
with st.spinner("Loading models and vector database..."):

    retriever = load_retriever()

    llm = load_llm()

st.success("System Loaded Successfully")


# User Input
query = st.text_input(
    "Enter your medical question:"
)


# Generate Response
if st.button("Generate Answer"):

    if query.strip() == "":

        st.warning(
            "Please enter a question."
        )

    else:

        with st.spinner(
            "Generating response..."
        ):

            start_time = time.time()

            response = generate_rag_response(
                query=query,
                retriever=retriever,
                llm=llm
            )

            end_time = time.time()

            latency = round(
                end_time - start_time,
                2
            )


        # Display Response
        st.subheader("Generated Answer")

        st.write(response)

        # Display Metrics
        st.subheader("Inference Metrics")

        st.write(
            f"Response Time: {latency} seconds"
        )
