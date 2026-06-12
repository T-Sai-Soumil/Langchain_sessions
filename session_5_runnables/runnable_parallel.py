from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task='text-generation'
)

model = ChatHuggingFace(llm=llm)

llm2 = HuggingFaceEndpoint(
    repo_id="katanemo/Arch-Router-1.5B",
    task='text-generation'
)

model2=ChatHuggingFace(llm=llm2)

parser=StrOutputParser()

prompt1=PromptTemplate(
    template="Generate a tweet about {topic}",
    input_variables=['topic']
)

prompt2=PromptTemplate(
    template='Generate a Linkedin post about {topic}',
    input_variables=['topic']
)

parallel_chain=RunnableParallel({
    'tweet': RunnableSequence(prompt1,model,parser),
    'linkedin': RunnableSequence(prompt2,model2,parser)
})

result=parallel_chain.invoke({'topic':'AI'})
print(result['tweet'])
print(result['linkedin'])