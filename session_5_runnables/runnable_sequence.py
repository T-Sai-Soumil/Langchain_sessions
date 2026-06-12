from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task='text-generation'
)

model=ChatHuggingFace(llm=llm)

prompt=PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

parser=StrOutputParser()

chain=RunnableSequence(prompt,model,parser)

print(chain.invoke({'topic':'AI'}))
