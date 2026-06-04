from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import Optional

load_dotenv()

class Review(BaseModel):
    key_themes: list[str] = Field(description="Write down all the key themes discussed in the review in a list")
    summary: str = Field(description="A brief summary of the user review.")
    sentiment: str = Field(description="The sentiment of the review (e.g., positive, negative, neutral).")
    pros: Optional[list[str]] = Field(description="Write down all the positives inside a list (null if none)")
    cons: Optional[list[str]] = Field(description="Write down all the negatives inside a list (null if none)")


llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-1B-Instruct",
    task='text-generation',
    temperature=0.01
)

model = ChatHuggingFace(llm=llm)

parser = JsonOutputParser(pydantic_object=Review)

prompt = PromptTemplate(
    template="""You are an assistant that analyzes product reviews. Respond ONLY with a valid JSON object matching the schema. Do not include any explanations.

Example:
Input: "The camera quality is amazing, but the speakers are quiet."
Output:
{{
  "key_themes": ["camera", "speakers"],
  "summary": "Great camera quality but the speakers are quiet.",
  "sentiment": "neutral",
  "pros": ["camera quality"],
  "cons": ["quiet speakers"]
}}

Instructions:
{format_instructions}

Review to analyze:
"{query}"
""",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

chain = prompt | model | parser

try:
    result = chain.invoke({
        "query": "Mobile is very slim, which feels great to use but the battery life is not good. Also the amount of bloatware is insane"
    })
    print("SUCCESSFUL PARSED OUTPUT:")
    print(result)
except Exception as e:
    import traceback
    traceback.print_exc()
