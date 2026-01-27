from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# 1. Setup Database
print("🧠 Loading Memory...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory="vectorstore/", embedding_function=embeddings)

# 2. Setup LLM (Pointing to your Local LM Studio)
llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
    model="medical-llama-3-8b",
    temperature=0.3
)

# 3. Create the "Retrieval Chain"
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_db.as_retriever(search_kwargs={"k": 2}), # Retrieve top 2 matching chunks
    return_source_documents=True
)

# 4. Ask a Question
question = "What are the common symptoms of diabetes?"
print(f"❓ Question: {question}")

response = qa_chain.invoke({"query": question})

print(f"🤖 Answer: {response['result']}")
print("\n📄 Source Document Used:")
print(response['source_documents'][0].page_content)