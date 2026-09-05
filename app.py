import os

import streamlit as st
import pymupdf
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
from google import genai


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Chat with PDF",
    page_icon="📄",
    layout="wide"
)


# =========================================================
# LOAD API KEY
# =========================================================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("GEMINI_API_KEY not found in .env file.")
    st.stop()

client = genai.Client(
    api_key=api_key
)


# =========================================================
# LOAD EMBEDDING MODEL
# =========================================================

@st.cache_resource
def load_embedding_model():

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    return model


model = load_embedding_model()


# =========================================================
# EXTRACT TEXT FROM PDF
# =========================================================

def extract_text(uploaded_file):

    # Read uploaded file as bytes
    pdf_bytes = uploaded_file.getvalue()

    # Open PDF from memory
    doc = pymupdf.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    text = ""

    for page in doc:

        text += page.get_text()

    doc.close()

    return text


# =========================================================
# SPLIT TEXT INTO CHUNKS
# =========================================================

def create_chunks(text):

    splitter = RecursiveCharacterTextSplitter(

        chunk_size=500,

        chunk_overlap=50
    )

    chunks = splitter.split_text(text)

    return chunks


# =========================================================
# CREATE FAISS INDEX
# =========================================================

def create_faiss_index(chunks):

    # Create embeddings
    embeddings = model.encode(chunks)

    # FAISS requires float32
    embeddings = embeddings.astype("float32")

    # Get embedding dimension
    dimension = embeddings.shape[1]

    # Create FAISS index
    index = faiss.IndexFlatL2(
        dimension
    )

    # Add vectors
    index.add(embeddings)

    return index, embeddings


# =========================================================
# RETRIEVE RELEVANT CHUNKS
# =========================================================

def retrieve_chunks(
    query,
    index,
    chunks,
    k=5
):

    k = min(
        k,
        len(chunks)
    )

    # Convert query into embedding
    query_embedding = model.encode(
        [query]
    )

    query_embedding = query_embedding.astype(
        "float32"
    )

    # Search FAISS
    distances, indices = index.search(
        query_embedding,
        k
    )

    results = []

    for distance, idx in zip(
        distances[0],
        indices[0]
    ):

        results.append({

            "chunk": chunks[idx],

            "distance": float(distance)

        })

    return results


# =========================================================
# GENERATE ANSWER USING GEMINI
# =========================================================

def generate_answer(
    query,
    results
):

    # Combine retrieved chunks
    context = "\n\n".join(

        result["chunk"]

        for result in results

    )

    # RAG prompt
    prompt = f"""
You are a helpful document question-answering assistant.

Answer the user's question using ONLY the information
provided in the document context below.

Do not use outside knowledge.

If the answer cannot be found in the provided context,
say:

"I don't know based on the provided document."

Document Context:
{context}

User Question:
{query}

Answer clearly and concisely.
"""

    response = client.models.generate_content(

        model="gemini-3.6-flash",

        contents=prompt
    )

    return response.text


# =========================================================
# SESSION STATE
# =========================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


if "chunks" not in st.session_state:

    st.session_state.chunks = None


if "index" not in st.session_state:

    st.session_state.index = None


if "embeddings" not in st.session_state:

    st.session_state.embeddings = None


if "file_name" not in st.session_state:

    st.session_state.file_name = None


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("📄 Chat with PDF")

    st.write(
        "Upload a PDF and ask questions about it."
    )

    st.divider()

    # ---------------------------------------------
    # FILE UPLOAD
    # ---------------------------------------------

    uploaded_file = st.file_uploader(

        "Upload your PDF",

        type=["pdf"]
    )

    st.divider()

    # ---------------------------------------------
    # RETRIEVAL SETTINGS
    # ---------------------------------------------

    st.subheader("⚙️ Retrieval Settings")

    k = st.slider(

        "Number of chunks",

        min_value=1,

        max_value=10,

        value=5
    )

    st.divider()

    # ---------------------------------------------
    # CLEAR CHAT
    # ---------------------------------------------

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()


# =========================================================
# PROCESS UPLOADED PDF
# =========================================================

if uploaded_file:

    # Check if this is a new file

    if (
        st.session_state.file_name
        != uploaded_file.name
    ):

        # Clear previous chat
        st.session_state.messages = []

        # Save file name
        st.session_state.file_name = (
            uploaded_file.name
        )

        # ---------------------------------------------
        # EXTRACT TEXT
        # ---------------------------------------------

        with st.spinner(
            "📖 Reading PDF..."
        ):

            text = extract_text(
                uploaded_file
            )

        # Check if text exists

        if not text.strip():

            st.error(
                "Could not extract text from this PDF."
            )

            st.stop()

        # ---------------------------------------------
        # CREATE CHUNKS
        # ---------------------------------------------

        with st.spinner(
            "✂️ Splitting document..."
        ):

            chunks = create_chunks(
                text
            )

        # ---------------------------------------------
        # CREATE EMBEDDINGS + FAISS
        # ---------------------------------------------

        with st.spinner(
            "🧠 Creating embeddings..."
        ):

            index, embeddings = (
                create_faiss_index(
                    chunks
                )
            )

        # Store everything in session state

        st.session_state.chunks = chunks

        st.session_state.index = index

        st.session_state.embeddings = embeddings

        st.success(
            "✅ PDF processed successfully!"
        )


# =========================================================
# MAIN HEADER
# =========================================================

st.title("📄 Chat with PDF")

st.write(
    "Upload a PDF and ask questions about its contents "
    "using Retrieval-Augmented Generation."
)


# =========================================================
# SHOW DOCUMENT INFORMATION
# =========================================================

if (
    st.session_state.chunks
    and st.session_state.embeddings is not None
):

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "📄 Document",
            st.session_state.file_name
        )

    with col2:

        st.metric(
            "🧩 Chunks",
            len(st.session_state.chunks)
        )

    with col3:

        st.metric(
            "🔢 Embedding Size",
            st.session_state.embeddings.shape[1]
        )

    st.divider()


# =========================================================
# WELCOME MESSAGE
# =========================================================

if not uploaded_file:

    st.info(
        "👋 Upload a PDF from the sidebar to get started."
    )

    st.markdown(
        """
        ### What can you ask?

        After uploading a document, you can ask questions such as:

        - What is this document about?
        - Summarize the main points.
        - What are the important topics?
        - Explain the methodology used.
        - What conclusions are mentioned?
        - Find information about a specific topic.
        """
    )


# =========================================================
# DISPLAY PREVIOUS CHAT
# =========================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.write(
            message["content"]
        )


# =========================================================
# CHAT INPUT
# =========================================================

if uploaded_file:

    query = st.chat_input(
        "Ask a question about your PDF..."
    )

    # ---------------------------------------------
    # PROCESS QUESTION
    # ---------------------------------------------

    if query:

        # Store user message

        st.session_state.messages.append({

            "role": "user",

            "content": query
        })

        # Display user message

        with st.chat_message("user"):

            st.write(query)

        # -----------------------------------------
        # ASSISTANT RESPONSE
        # -----------------------------------------

        with st.chat_message("assistant"):

            # Retrieve chunks

            with st.spinner(
                "🔎 Searching the document..."
            ):

                results = retrieve_chunks(

                    query,

                    st.session_state.index,

                    st.session_state.chunks,

                    k
                )

            # Generate answer

            with st.spinner(
                "🤖 Generating answer..."
            ):

                answer = generate_answer(

                    query,

                    results
                )

            # Display answer

            st.write(answer)

            # -----------------------------------------
            # SHOW RETRIEVED CONTEXT
            # -----------------------------------------

            with st.expander(
                "🔎 View Retrieved Context"
            ):

                for i, result in enumerate(
                    results,
                    start=1
                ):

                    st.markdown(
                        f"**Chunk {i}**"
                    )

                    st.write(
                        result["chunk"]
                    )

                    st.caption(
                        f"Distance: "
                        f"{result['distance']:.4f}"
                    )

                    st.divider()

        # Store assistant response

        st.session_state.messages.append({

            "role": "assistant",

            "content": answer
        })