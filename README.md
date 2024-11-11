# Document Embedding and Retrieval API

This project provides an API to embed and retrieve information from multiple document formats (`.pdf`, `.docx`, and `.txt`). It leverages the LangChain library with FAISS for efficient vector search, Google Generative AI embeddings for high-quality document embeddings, and Groq-powered LLMs to generate answers to queries based on document content.

## Setup Instructions

1. **Clone the repository**:
    git clone <repo>
    cd <repo>

2. **Set up virtual environment** :
    python -m venv env
    env\Scripts\activate  # For Windows

3. **Install dependencies**:
    pip install -r requirements.txt

4. **Environment variables**: 
   - Create a `.env` file in the project root and add your API keys:
     GROQ_API_KEY=your_groq_api_key
     GOOGLE_API_KEY=your_google_api_key

5. **Run the FastAPI server**:
  
    uvicorn app:app --reload

    - The server will run at `http://127.0.0.1:8000`.

## API Documentation

### 1. Embedding Endpoint

**Endpoint**: `POST /api/embedding`

**Description**: This endpoint processes a list of documents, extracts their content, generates embeddings, and stores them in a FAISS vector index for future retrieval.

### 2. Query Endpoint

**Endpoint**: `POST /api/embedding`

**Description**: This endpoint processes query generates embeddings,macthes it with vector store and give answer.

## Technology Choice Justifications

- **LangChain**: Provides a robust framework for working with language models and simplifies creating retrieval-augmented generation (RAG) setups.
- **FAISS**: A powerful library for efficient vector similarity search, essential for handling document embeddings in large quantities.We can use chromadb also but used this beacuse i have used it earlier

- **Google Generative AI Embeddings**: Offers high-quality document embeddings, crucial for accurate document retrieval.we can use hugging face embeddings also in this place

- **Groq LLM(LLAMA-70b-versatile)**: A flexible LLM that provides tailored answers to queries, allowing more coherent responses based on document content and we can use hugging face opensource llm model also but used this for fast inference purpose 
- **Reranker LLM** : In reranking i have used the llm the rank the documents retrieved and than a llm to give response
