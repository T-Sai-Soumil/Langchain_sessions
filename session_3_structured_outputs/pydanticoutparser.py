from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task='text-generation'
)

model = ChatHuggingFace(llm=llm)

class Person(BaseModel):
    name: str=Field(description='Name of the person')
    age: int=Field(gt=18,description='Age of the person')
    city: str=Field(description='Name of the city the person belongs to')

parser=PydanticOutputParser(pydantic_object=Person)

template=PromptTemplate(
    template='Generate the name, age and city of a fictional {topic} character \n {format_instructions}',
    input_variables=['place'],
    partial_variables={'format_instructions':parser.get_format_instructions()}
)

# prompt=template.invoke({'topic':'anime'})
# print(prompt)

chain=template|model|parser
result=chain.invoke({'topic':'anime'})
print(result)
print(type(result))