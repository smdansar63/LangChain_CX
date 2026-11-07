from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()

model = ChatGroq(model = "openai/gpt-oss-20b")

prompt = PromptTemplate(
    template = "what is this webpage about? {text}",
    input_variables=['text']
)

parser = StrOutputParser()

loader = WebBaseLoader("https://www.nasa.gov/")

docs = loader.load()

chain = prompt | model | parser

result = chain.invoke({
    'text': docs[0].page_content[:3000]
})

print(result)