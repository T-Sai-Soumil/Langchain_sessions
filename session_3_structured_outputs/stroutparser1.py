from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-1B-Instruct",
    task='text-generation',
    temperature=0.01
)

model = ChatHuggingFace(llm=llm)

template1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variabls=['topic']
)

template2 = PromptTemplate(
    template='Write a 5 line summary on the following text. /n {text}',
    input_variabls=['text']
)

parser=StrOutputParser()

chain=template1|model|parser|template2|model|parser

result=chain.invoke({'topic':'black hole'})

print(result)