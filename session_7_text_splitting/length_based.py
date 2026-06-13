from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader=PyPDFLoader('Sriniously-Backend-Guide-Enhanced.pdf')

docs=loader.load()

text="""
Space exploration expands humanity's understanding of the universe beyond Earth.
Scientists use satellites, probes, and spacecraft to study distant planets and stars.
Missions to space have led to many technological advancements that benefit daily life.
Exploring other worlds may help us find signs of life beyond our planet.
As technology improves, space exploration continues to inspire curiosity and innovation.
"""

splitter=CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
    separator='',
)

# result=splitter.split_text(text)
result=splitter.split_documents(docs)

print(result[0])