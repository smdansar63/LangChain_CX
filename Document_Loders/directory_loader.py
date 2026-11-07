from langchain_community.document_loaders import PyPDFLoader , DirectoryLoader

loader = DirectoryLoader(
    path = "Sample_Documents", 
    glob='**/*.pdf', 
    loader_cls=PyPDFLoader
)

docs = loader.load()

# lazy load

# docs = loader.lazy_load()

for doc in docs:
    print(doc.page_content)
    print(doc.metadata)