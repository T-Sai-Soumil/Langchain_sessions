from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader=DirectoryLoader(
    path='books',
    glob='*.pdf',
    loader_cls=PyPDFLoader
)

# docs=loader.load() = dumps everything at once in the memory when ready

docs=loader.lazy_load()  # loads one at a time in memory 

for document in docs:
    print(document.metadata)