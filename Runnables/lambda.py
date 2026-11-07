from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence , RunnableLambda , RunnableParallel , RunnablePassthrough

load_dotenv()

def word_count(text: str) -> int:
    return len(text.split())

prompt1 = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

model = ChatGroq(model = "openai/gpt-oss-20b")

parser = StrOutputParser()


joke_chain = RunnableSequence(prompt1, model, parser)

parallel_chain = RunnableParallel({
    "joke" : RunnablePassthrough(),
    "word_count" : RunnableLambda(word_count)
}
)

chain = RunnableSequence(joke_chain, parallel_chain)


result = chain.invoke({'topic':'AI'})

final_result = """{} \n word count - {}""".format(result['joke'], result['word_count'])

print(final_result)