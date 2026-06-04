from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from typing import TypedDict

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-1B-Instruct",
    task='text-generation'
)

model = ChatHuggingFace(llm=llm)

class Review(TypedDict):
    summary: str
    sentiment: str

structured_model = model.with_structured_output(Review)    

try:
    result = structured_model.invoke(
        "Mobile is very slim, which feels great to use but the battery life is not good. Also the amount of bloatware is insane"
    )
    print(result)

except Exception as e:
    print(type(e))
    print(e)