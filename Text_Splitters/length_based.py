from langchain_text_splitters import CharacterTextSplitter

text = "This is a sample text that we will split into smaller chunks based on character length."

splitter = CharacterTextSplitter(chunk_size=20, chunk_overlap=5 , separator='')

chunks = splitter.split_text(text)

print(len(chunks))
print(chunks)


from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('sample.pdf')

doc_chunks = loader.load()

doc_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)    

result = doc_splitter.split_documents(doc_chunks)

print(len(result))

print(result[0].page_content)