# 📄 Chat with PDF — RAG PDF Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that lets you upload any PDF and ask natural-language questions about its contents. It retrieves the most relevant chunks of the document using semantic search and generates grounded answers using Google's Gemini API — so answers come from your document, not the model's general knowledge.

**Live demo:** _https://yuvraj-pdf-rag.streamlit.app/_

---

## ✨ Features

- 📤 Upload any PDF and extract its text automatically
- ✂️ Smart chunking of document text for better retrieval
- 🧠 Semantic search using sentence-transformer embeddings + FAISS
- 🤖 Context-grounded answers via Google Gemini (`gemini-3.6-flash`)
- 💬 Persistent chat history within a session
- 🔎 Expandable view of retrieved context chunks (with similarity distance) for transparency
- ⚙️ Adjustable number of retrieved chunks (top-k) via sidebar slider
- 🗑️ One-click chat clearing

---

## 🏗️ How It Works

1. **Extract** — The uploaded PDF is parsed page-by-page using PyMuPDF to extract raw text.
2. **Chunk** — Text is split into overlapping chunks (500 chars, 50 overlap) using LangChain's `RecursiveCharacterTextSplitter`.
3. **Embed** — Each chunk is converted into a vector embedding using the `all-MiniLM-L6-v2` SentenceTransformer model.
4. **Index** — Embeddings are stored in a FAISS `IndexFlatL2` index for fast similarity search.
5. **Retrieve** — When a question is asked, it's embedded and the top-k most similar chunks are retrieved from the index.
6. **Generate** — The retrieved chunks are passed as context to Gemini, which generates an answer grounded strictly in that context.

```
PDF Upload → Text Extraction → Chunking → Embeddings → FAISS Index
                                                              │
User Question ──────────────► Embed Query ──► Similarity Search
                                                              │
                                              Retrieved Chunks + Question
                                                              │
                                                     Gemini API (RAG prompt)
                                                              │
                                                          Answer
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| UI / App framework | [Streamlit](https://streamlit.io/) |
| PDF parsing | [PyMuPDF](https://pymupdf.readthedocs.io/) |
| Text chunking | [LangChain Text Splitters](https://python.langchain.com/) |
| Embeddings | [Sentence Transformers](https://www.sbert.net/) (`all-MiniLM-L6-v2`) |
| Vector search | [FAISS](https://github.com/facebookresearch/faiss) |
| LLM | [Google Gemini API](https://ai.google.dev/) (`google-genai`) |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- A [Gemini API key](https://aistudio.google.com/apikey)

### Installation

```bash
git clone https://github.com/YuvrajManhas/rag-pdf-chatbot.git
cd rag-pdf-chatbot
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root (see `.env.example`):

```
GEMINI_API_KEY=your_api_key_here
```

If deploying on **Streamlit Community Cloud**, add the same key under your app's **Settings → Secrets** instead:

```toml
GEMINI_API_KEY = "your_api_key_here"
```

### Run locally

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## 📖 Usage

1. Upload a PDF from the sidebar.
2. Wait for it to be chunked and embedded (progress shown via spinners).
3. Ask a question in the chat input, e.g.:
   - "What is this document about?"
   - "Summarize the main points."
   - "What methodology is used?"
4. Expand **🔎 View Retrieved Context** under any answer to see exactly which chunks were used to generate it.
5. Adjust the **Number of chunks** slider in the sidebar to control how much context is retrieved per question.

---

## 📂 Project Structure

```
rag-pdf-chatbot/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── .env.example         # Sample environment variable file
├── .gitignore
└── README.md
```

---

## ⚠️ Known Limitations

- Embeddings and the FAISS index are rebuilt from scratch for every new PDF and reset on session end (no persistence across restarts).
- Scanned/image-only PDFs without an embedded text layer won't extract any text (no OCR fallback yet).
- Answer quality depends on chunking granularity — very large or poorly structured PDFs may need tuned chunk size/overlap.

---

## 🗺️ Roadmap / Ideas

- [ ] OCR fallback for scanned PDFs
- [ ] Persistent vector store (e.g. Chroma/Pinecone) across sessions
- [ ] Multi-document support
- [ ] Source page number citations in answers
- [ ] Streaming responses

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome. Feel free to check the [issues page](https://github.com/YuvrajManhas/rag-pdf-chatbot/issues).

## 👤 Author

**Yuvraj Manhas**
GitHub: [@YuvrajManhas](https://github.com/YuvrajManhas)
