from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser , StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field 
from typing import Literal

load_dotenv()

model = ChatGroq(model = "openai/gpt-oss-20b")


class Feedback(BaseModel):

    sentiment: Literal['positive', 'negative'] = Field(description='Give the sentiment of the feedback')

parser2 = PydanticOutputParser(pydantic_object=Feedback)

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template='Classify the sentiment of the following feedback text into postive or negative \n {feedback} \n {format_instruction}',
    input_variables=['feedback'],
    partial_variables={'format_instruction':parser2.get_format_instructions()}
)

classify_chain = prompt1 | model | parser2

prompt2 = PromptTemplate(
    template='Write an appropriate response to this positive feedback \n {feedback}',
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template='Write an appropriate response to this negative feedback \n {feedback}',
    input_variables=['feedback']
)

branch_chain = RunnableBranch(
     (lambda x: x.sentiment == 'positive', prompt2 | model | parser),
     (lambda x: x.sentiment == 'negative', prompt3 | model | parser),
     RunnableLambda(lambda x: "Invalid sentiment")
)

chain = classify_chain | branch_chain

result = chain.invoke({"feedback": "The product is amazing and exceeded my expectations!"})

print(result)