from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# We treat LM Studio like it's OpenAI, but point it to localhost
local_llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",  # Not needed locally, but required by the library
    model="medical-llama-3-8b", # The name doesn't strictly matter for local
    temperature=0.7
)

print("🔌 Connecting to Local AI Server...")

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Are you working? Reply with 'Yes, I am online!'"),
]

response = local_llm.invoke(messages)

print(f"🤖 AI Response: {response.content}")