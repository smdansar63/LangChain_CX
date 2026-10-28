from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel , Field

load_dotenv()

model = ChatGroq(model = "openai/gpt-oss-20b")

class Details(BaseModel):
    name : str = Field(description = "The name of the person or entity being described")
    age : int = Field(gt = 18 , description = "The age of the person or entity being described, must be greater than 18")
    place : str = Field(description = "The place where the person or entity is located")

parser = PydanticOutputParser(pydantic_object=Details)

template = PromptTemplate(
    template='Generate the name, age and city of a fictional {place} person \n {format_instruction}',
    input_variables=['place'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({"place": "New York"})

print(result)