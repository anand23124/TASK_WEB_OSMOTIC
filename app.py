from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import List
import os
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate       
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import tempfile
import shutil

app = FastAPI()

# Load environment variables
load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')
google_api_key = os.getenv('GOOGLE_API_KEY')

# Initialize embeddings and FAISS vector store
embeddings = GoogleGenerativeAIEmbeddings(google_api_key=google_api_key, model="models/embedding-001")
llm = ChatGroq(temperature=0, groq_api_key=groq_api_key, model_name="llama-3.1-70b-versatile")

# Endpoint to generate document embeddings
@app.post("/api/embedding")
async def embed_documents(documents: List[UploadFile] = File(...)):
    text = ""

    try:
        # Process each uploaded document
        for document in documents:
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                shutil.copyfileobj(document.file, temp_file)
                temp_file_path = temp_file.name

            # Load and process document based on file type
            if document.filename.endswith(".pdf"):
                loader = PyPDFLoader(temp_file_path, extract_images=True)
            elif document.filename.endswith(".docx"):
                loader = Docx2txtLoader(temp_file_path)
            elif document.filename.endswith(".txt"):
                loader = TextLoader(temp_file_path)
            else:
                raise HTTPException(status_code=400, detail="Unsupported file type")

            # Extract and append the content
            docs = loader.load()
            text += "".join([doc.page_content for doc in docs])

            # Remove temporary file
            os.remove(temp_file_path)

        # Split and vectorize the combined text
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        cleaned_text = text_splitter.split_text(text)
        vectors = FAISS.from_texts(cleaned_text, embeddings)
        
        # Save the FAISS index locally for retrieval
        vectors.save_local("faiss_index")

        return {"message": "Documents embedded successfully."}
    except Exception as e:
        return {"error": f"Failed to embed documents: {str(e)}"}
# Endpoint to query based on embeddings
@app.post("/api/query")
async def query_document(query: str):
    try:
        # Load the FAISS vector store
        vector = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        retriever = vector.as_retriever()

        # Set up the chain with a prompt template and the retriever
        prompt = ChatPromptTemplate.from_template(
            """
            This context is about the capabilities of a company
            <context>
            {context}
            <context>
            Questions: {input}
            Provide a summary of this whole context.
            """
        )
        document_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        # Perform the retrieval and get the response
        response = retrieval_chain.invoke({'input': query})
        answer = response['answer']

        return {"response": answer}
    except Exception as e:
        return {"error": f"Please start a new session. Error: {str(e)}"}
