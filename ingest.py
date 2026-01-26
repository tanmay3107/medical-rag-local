import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 1. Settings
DATA_PATH = "data/"
DB_PATH = "vectorstore/"

def create_vector_db():
    print("📄 Loading PDF...")
    # Load all PDFs in the data folder
    documents = []
    for file in os.listdir(DATA_PATH):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(DATA_PATH, file))
            documents.extend(loader.load())

    print(f"   Loaded {len(documents)} pages.")

    # 2. Split Text
    print("✂️  Splitting text...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,    # How big each chunk is
        chunk_overlap=50   # Overlap to keep context between chunks
    )
    chunks = text_splitter.split_documents(documents)
    print(f"   Created {len(chunks)} chunks.")

    # 3. Embed & Store
    print("💾 Creating Vector Store (This may take a moment)...")
    
    # We use a small, fast model for embeddings (runs on CPU easily)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create and persist the database
    vector_db = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory=DB_PATH
    )
    
    print("✅ Vector Database Created Successfully!")

if __name__ == "__main__":
    create_vector_db()