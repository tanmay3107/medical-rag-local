from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOpenAI
# ✅ MODERN IMPORTS
from langchain_community.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 1. Setup Database
print("🧠 Loading Memory...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory="vectorstore/", embedding_function=embeddings)
retriever = vector_db.as_retriever(search_kwargs={"k": 2}) # Get top 2 matches

# 2. Setup LLM (Pointing to your Local LM Studio)
llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
    model="medical-llama-3-8b",
    temperature=0.3
)

# 3. Define the Prompt (Your snippet)
system_prompt = (
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, say that you don't know. "
    "Answer concisely and professionally."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# 4. Create the Modern Chains (Your snippet)
print("🔗 Building Retrieval Chain...")
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# 5. Ask a Question
question = "What are the common symptoms of diabetes?"
print(f"❓ Question: {question}")

response = rag_chain.invoke({"input": question})

print(f"🤖 Answer: {response['answer']}")
print("\n📄 Source Document Used:")
# In the new chain, sources are under 'context', not 'source_documents'
print(response['context'][0].page_content)