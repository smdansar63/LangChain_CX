from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt1 = PromptTemplate(
    template = "Which team won IPL in year /n {year}?",
    input_variables = ["year"]
)

prompt2 = PromptTemplate(
    template = "Who was the captain of the {text}?",
    input_variables = ["text"]
)

model = ChatGroq(model = "openai/gpt-oss-20b")

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({"year": "2023"})

print(result)