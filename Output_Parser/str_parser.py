from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGroq(model = "openai/gpt-oss-20b")


template1 = PromptTemplate(
    template = "Write a detailed report about {topic}.",
    input_variables = ["topic"]
    )

template2 = PromptTemplate(
    template = "Write a 5 line summary about this report {text}.",
    input_variables = ["text"]
    )

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({"topic": "Are any topics from AI engineering like llms , rag , agents , gen ai related to Samsung Prism?"})

print(result)