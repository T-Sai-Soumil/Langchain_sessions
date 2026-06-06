from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task='text-generation'
)

model=ChatHuggingFace(llm=llm)

class Feedback(BaseModel):
    sentiment: Literal['positive','negative'] = Field(description='Give the sentiment of the feedback')

parser=PydanticOutputParser(pydantic_object=Feedback)    

prompt1=PromptTemplate(
    template='Classify the sentiment of the following feedback text into positive or negative \n {feedback} \n {format_instructions}',
    input_variables=['feedback'],
    partial_variables={'format_instructions':parser.get_format_instructions()}
)

classifier_chain=prompt1|model|parser

# result=classifier_chain.invoke({'feedback':'This is a wonderful smartphone'}).sentiment
# print(result)

prompt2=PromptTemplate(
    template='Write an appropriate response to this positive feedback \n {feedback}',
    input_variables=['feedback']
)

prompt3=PromptTemplate(
    template='Write an appropriate response to this negative feedback \n {feedback}',
    input_variables=['feedback']
)

parser1=StrOutputParser()

branch_chain=RunnableBranch(
    (lambda x:x.sentiment == 'positive', prompt2|model|parser1),
    (lambda x:x.sentiment == 'negative', prompt3|model|parser1),
    RunnableLambda(lambda x:"could not find sentiment")
)

chain=classifier_chain|branch_chain
result=chain.invoke({'feedback':'This is a terrible phone'})
print(result)
