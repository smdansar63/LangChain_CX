from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq( model = "openai/gpt-oss-20b")

prompt_template = PromptTemplate(
    template = "What is the highest score of this cricketer in ODI? {cricketer_name}",
    input_variables = ["cricketer_name"]
)

prompt = prompt_template.invoke({"cricketer_name": "Sachin Tendulkar"})

response = model.invoke(prompt)

print(response.content)