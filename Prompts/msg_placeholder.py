from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq( model = "openai/gpt-oss-20b")

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{user_input}")
])

chat_history = []

with open('chathistory.txt') as f:
    chat_history.extend(f.readlines())

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Exiting the chat...")
        break
    prompt = chat_prompt.invoke({"chat_history": chat_history, "user_input": user_input})
    chat_history.append(HumanMessage(content=user_input))
    response = model.invoke(prompt)
    chat_history.append(AIMessage(content=response.content))
    print("AI:", response.content)
    with open("chathistory.txt", "a") as f:
        f.write(f"\nHumanMessage(content={user_input!r})\n")
        f.write(f"AIMessage(content={response.content!r})")
