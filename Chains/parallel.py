from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

model = ChatGroq(model = "openai/gpt-oss-20b")  

prompt1 = PromptTemplate(
    template = "Give top 5 interview advice for role of {role} in {company}.",
    input_variables = ["role", "company"]
)

prompt2 = PromptTemplate(
    template = "Give top 5 interview questions for role of {role} in {company}.",
    input_variables = ["role", "company"]
)

prompt3 = PromptTemplate(
    template = " Merge the above two outputs {advice} and {questions} .",
    input_variables = ["advice", "questions"]
)

parser = StrOutputParser()

parallel_chain = RunnableParallel(
    advice = prompt1 | model | parser,
    questions = prompt2 | model | parser
)

merge_chain = prompt3 | model | parser

chain = parallel_chain | merge_chain

result = chain.invoke({"role": "AI Engineer", "company": "Google"})

print(result)