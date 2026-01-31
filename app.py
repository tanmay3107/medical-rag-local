import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 1. Page Config
st.set_page_config(page_title="Medical RAG AI", page_icon="🏥")
st.title("🏥 Private Medical AI Assistant")

# 2. Sidebar for Upload
with st.sidebar:
    st.header("Upload Medical Record")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    
    if uploaded_file:
        # Save file locally
        save_path = os.path.join("data", uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Loaded: {uploaded_file.name}")
        
        # Trigger Ingestion
        if st.button("🧠 Ingest & Memorize"):
            with st.spinner("Reading and embedding..."):
                loader = PyPDFLoader(save_path)
                docs = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                chunks = text_splitter.split_documents(docs)
                
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                # Create/Update Vector DB
                vector_db = Chroma.from_documents(
                    documents=chunks, 
                    embedding=embeddings, 
                    persist_directory="vectorstore/"
                )
                st.success("✅ Knowledge Updated!")

# 3. Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 4. Chat Logic
if prompt_text := st.chat_input("Ask a medical question..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt_text})
    with st.chat_message("user"):
        st.markdown(prompt_text)

    # RAG Generation
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        # Connect to Database
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_db = Chroma(persist_directory="vectorstore/", embedding_function=embeddings)
        retriever = vector_db.as_retriever(search_kwargs={"k": 2})
        
        # Connect to Local LLM
        llm = ChatOpenAI(
            base_url="http://localhost:1234/v1", 
            api_key="lm-studio", 
            model="medical-llama-3-8b", 
            temperature=0.3
        )

        # Build Chain
        template = """Answer the question based only on the following context:
        {context}
        
        Question: {question}
        """
        prompt = ChatPromptTemplate.from_template(template)
        
        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        # Run
        response = rag_chain.invoke(prompt_text)
        message_placeholder.markdown(response)
        
    # Add AI response to history
    st.session_state.messages.append({"role": "assistant", "content": response})