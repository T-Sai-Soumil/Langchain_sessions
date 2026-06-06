from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task='text-generation'
)

model=ChatHuggingFace(llm=llm)

template=PromptTemplate(
    template='Generate 5 interesting facts about {topic}',
    input_variables=['topic']
)

parser=StrOutputParser()
chain=template|model|parser
result=chain.invoke({'topic':'chess'})
print(result)
chain.get_graph().print_ascii()