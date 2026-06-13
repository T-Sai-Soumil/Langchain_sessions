from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

text="""
Space exploration expands humanity's understanding of the universe beyond Earth.
Scientists use satellites, probes, and spacecraft to study distant planets and stars.
Missions to space have led to many technological advancements that benefit daily life.
Exploring other worlds may help us find signs of life beyond our planet.
As technology improves, space exploration continues to inspire curiosity and innovation.
"""

splitter=RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
)

chunks=splitter.split_text(text)

print(len(chunks))
print(chunks)