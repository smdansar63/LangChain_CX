from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv
load_dotenv()

docs = [
    Document(page_content="LangChain helps developers build LLM applications easily."),
    Document(page_content="Chroma is a vector database optimized for LLM-based search."),
    Document(page_content="Embeddings convert text into high-dimensional vectors."),
    Document(page_content="OpenAI provides powerful embedding models."),
]

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

vector_store = Chroma(
    collection_name="langchain_docs",
    embedding_function=embeddings
)

vector_store.add_documents(docs)

retriever = vector_store.as_retriever(search_kwargs={"k": 2})

query = "What is LangChain?"

result = retriever.invoke(query)

for i, doc in enumerate(result):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)
