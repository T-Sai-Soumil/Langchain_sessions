from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

doc1=Document(
    page_content="Virat Kohli is one of the greatest modern-day cricketers, renowned for his exceptional batting technique, consistency, and passion for the game. He has captained India across formats and holds numerous records in international cricket.",
    metadata={"team":"Roysl Challengers Banglore"}
)

doc2=Document(
    page_content="Rohit Sharma is an elegant opening batsman known for his effortless stroke play and ability to score big hundreds. He is the only player to have scored three double centuries in One Day Internationals.",
    metadata={"team":"Mumbai Indians"}
)

doc3=Document(
    page_content="Jasprit Bumrah is India's premier fast bowler, famous for his unique bowling action, deadly yorkers, and accuracy under pressure. He has been a match-winner for India across all formats of the game.",
    metadata={"team":"Mumbai Indians"}
)

doc4=Document(
    page_content="MS Dhoni is one of cricket's most successful captains, celebrated for his calm leadership, lightning-fast wicketkeeping, and finishing abilities. He led India to victories in the 2007 T20 World Cup, 2011 Cricket World Cup, and 2013 Champions Trophy.",
    metadata={"team":"Chennai Super Kings"}
)

doc5=Document(
    page_content="Ravindra Jadeja is a world-class all-rounder known for his sharp fielding, economical left-arm spin, and valuable batting contributions. His versatility makes him one of the most complete cricketers in modern cricket.",
    metadata={"team":"Gujarat Titans"}
)

docs=[doc1,doc2,doc3,doc4,doc5]

vector_store=Chroma(
    embedding_function=HuggingFaceEmbeddings(),
    persist_directory='chroma_db',
    collection_name='sample'
)

vector_store.add_documents(docs)

data=vector_store.get(include=['embeddings','documents','metadatas'])

result1=vector_store.similarity_search(
    query='Who among these are a bowler',
    k=2
)

result2=vector_store.similarity_search_with_score(
    query='Who among these are a bowler',
    k=2
)

result3=vector_store.similarity_search_with_score(
    query="",
    filter={"team":"Mumbai Indians"}
)

updated_doc1=Document(
    page_content="Virat Kohli, at least according to some fans on social media is, that he can occasionally be overly intense or theatrical in his celebrations and on-field reactions. His aggressive send-offs, animated gestures, and emotional displays sometimes become memes and are seen as excessive by critics.",
    metadata={"team":"Royal Challengers Banglore"}
)

vector_store.update_document(document_id=data["ids"][0],document=updated_doc1)

vector_store.delete(ids=data["ids"][1])



