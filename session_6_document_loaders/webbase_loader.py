from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task='text-generation'
)

model = ChatHuggingFace(llm=llm)

parser=StrOutputParser()

prompt=PromptTemplate(
    template='Write a 1 line summary on the following question \n {question} from the following text -\n {text}',
    input_variables=['question','text']
)

url='https://www.amazon.in/Berserk-Deluxe-1-Kentaro-Miura/dp/1506711987'

loader=WebBaseLoader(url)

docs=loader.load()

# print(len(docs))

# print(docs[0].page_content)

chain=prompt|model|parser

print(chain.invoke({'question':'Which book is this page referring to?','text':docs[0].page_content}))