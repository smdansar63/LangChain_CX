from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader

load_dotenv()

model = ChatGroq(model = "openai/gpt-oss-20b")

prompt = PromptTemplate(
    template='What are the topics that are mentioned here just names /n {text}',
    input_variables= ['text']
)

parser = StrOutputParser()

loader = TextLoader('cricket.txt' , encoding='utf-8')

docs = loader.load()

chain = prompt | model | parser

result = chain.invoke({'text': docs[0].page_content})

print(result)