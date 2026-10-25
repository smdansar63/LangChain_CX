from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq( model = "openai/gpt-oss-20b")

response = model.invoke("What do you know about Samsung Prism?")

print(response.content)