from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOpenAI

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

question = "What are the common symptoms of diabetes?"
# 1. Load Vector DB
print("🧠 Loading Memory...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_db = Chroma(
    persist_directory="vectorstore/",
    embedding_function=embeddings
)

retriever = vector_db.as_retriever(search_kwargs={"k": 2})

print("\n🔎 Retrieved documents:")
docs = retriever.invoke(question)

if not docs:
    print("⚠️ No documents retrieved!")
else:
    for i, d in enumerate(docs):
        print(f"\n--- Document {i+1} ---")
        print(d.page_content[:500])

# 2. Local LLM (LM Studio)
llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
    model="medical-llama-3-8b",
    temperature=0.3
)

# 3. Prompt
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Using the following context, summarize the answer to the user's question "
            "in your own words. Be concise and medically accurate.\n\n{context}"
        ),
        ("human", "{question}")
    ]
)


# 4. Build RAG chain (PURE LCEL)
rag_chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# 5. Ask question
print(f"❓ Question: {question}")

answer = rag_chain.invoke(question)

print(f"\n🤖 Answer:\n{answer}")
