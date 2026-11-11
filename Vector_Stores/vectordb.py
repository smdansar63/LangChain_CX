from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

doc1 = Document(
        page_content="Virat Kohli is one of the most successful and consistent batsmen in IPL history. Known for his aggressive batting style and fitness, he has led the Royal Challengers Bangalore in multiple seasons.",
        metadata={"team": "Royal Challengers Bangalore"}
    )
doc2 = Document(
        page_content="Rohit Sharma is the most successful captain in IPL history, leading Mumbai Indians to five titles. He's known for his calm demeanor and ability to play big innings under pressure.",
        metadata={"team": "Mumbai Indians"}
    )
doc3 = Document(
        page_content="MS Dhoni, famously known as Captain Cool, has led Chennai Super Kings to multiple IPL titles. His finishing skills, wicketkeeping, and leadership are legendary.",
        metadata={"team": "Chennai Super Kings"}
    )
doc4 = Document(
        page_content="Jasprit Bumrah is considered one of the best fast bowlers in T20 cricket. Playing for Mumbai Indians, he is known for his yorkers and death-over expertise.",
        metadata={"team": "Mumbai Indians"}
    )
doc5 = Document(
        page_content="Ravindra Jadeja is a dynamic all-rounder who contributes with both bat and ball. Representing Chennai Super Kings, his quick fielding and match-winning performances make him a key player.",
        metadata={"team": "Chennai Super Kings"}
    )

docs = [doc1, doc2, doc3, doc4, doc5]

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

vector_store = Chroma(
    collection_name="ipl_players",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

# add documents to the vector store

vector_store.add_documents(docs,ids=["1", "2", "3", "4", "5"])

# view documents in the vector store

vector_store.get(include=["embeddings", "metadatas", "documents"])

# search for documents in the vector store

vector_store.similarity_search("Who among these are bowlers?", k=2)

# search with similarity score

vector_store.similarity_search("Who among these are batsmen?", k=2)

# meta-data filtering

vector_store.similarity_search_with_score(
    query = "Who plays for Mumbai Indians?",
    filter = {"team": "Mumbai Indians"},
)

#update documents in the vector store

updated_doc1 = Document(
    page_content="Virat Kohli, the former captain of Royal Challengers Bangalore (RCB), is renowned for his aggressive leadership and consistent batting performances. He holds the record for the most runs in IPL history, including multiple centuries in a single season. Despite RCB not winning an IPL title under his captaincy, Kohli's passion and fitness set a benchmark for the league. His ability to chase targets and anchor innings has made him one of the most dependable players in T20 cricket.",
    metadata={"team": "Royal Challengers Bangalore"}
)

vector_store.update_document(document_id='1', document=updated_doc1)
1
# delete documents from the vector store

vector_store.delete(ids=["2"])

print(vector_store.get(include=['embeddings','documents', 'metadatas']))