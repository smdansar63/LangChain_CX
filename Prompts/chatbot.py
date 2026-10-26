from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

model = ChatGroq( model = "openai/gpt-oss-20b")

chat_history = [
    SystemMessage(content="You are a helpful assistant.")
]

while True:
    user_input = input("User :")
    if user_input.lower() in ["exit", "quit"]:
        print("Exiting the chat...")
        break
    chat_history.append(HumanMessage(content=user_input))
    response = model.invoke(chat_history)
    chat_history.append(AIMessage(content=response.content))
    print("AI :", response.content)