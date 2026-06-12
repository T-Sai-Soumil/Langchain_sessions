from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough
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
    template="Generate a joke about {topic}",
    input_variables=['topic']
)

prompt2=PromptTemplate(
    template='Generate an explanation for the following joke - {text}',
    input_variables=['text']
)

joke_gen_chain = RunnableSequence(prompt1,model,parser)

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'explanation': RunnableSequence(prompt2,model,parser)
})

final_chain=RunnableSequence(joke_gen_chain,parallel_chain)

result=final_chain.invoke({'topic':'cricket'})
print(result)