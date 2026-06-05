from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task='text-generation'
)

model = ChatHuggingFace(llm=llm)

parser = JsonOutputParser()

template=PromptTemplate(
    template='Give me 5 facts about black holes {topic} \n {format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

chain=template|model|parser
final_result=chain.invoke({'topic':'black_hole'})
print(final_result)
print(type(final_result)) # python treats json objects as dict
