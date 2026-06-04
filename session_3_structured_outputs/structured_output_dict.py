from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-1B-Instruct",
    task='text-generation',
    temperature=0.01
)

model = ChatHuggingFace(llm=llm)

# No Pydantic class needed — we use a plain JsonOutputParser
parser = JsonOutputParser()

prompt = PromptTemplate(
    template="""You are an assistant that analyzes product reviews. Respond ONLY with a valid JSON object matching the schema. Do not include any explanations.

The JSON object must have these keys:
- "key_themes": a list of strings — all key themes discussed in the review
- "summary": a string — a brief summary of the review
- "sentiment": a string — the sentiment (positive, negative, or neutral)
- "pros": a list of strings or null — all the positives
- "cons": a list of strings or null — all the negatives

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

Review to analyze:
"{query}"
""",
    input_variables=["query"],
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
