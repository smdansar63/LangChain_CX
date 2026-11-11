import wikipedia
from langchain_community.retrievers.wikipedia import WikipediaRetriever

retriever = WikipediaRetriever(
    top_k=2,
    lang="en"
)

query = "What is the Indian Premier League (IPL)?"

result = retriever.invoke(query)

for i, doc in enumerate(result):
    print(f"\n--- Result {i+1} ---")
    print("Document Content:", doc.page_content)
    print("Document Metadata:", doc.metadata)
    print("-----")