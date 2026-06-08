from fake_PromptTemplate import fakePromptTemplate
from fake_LLM import fakeLLM


class fakeLLMChain:
    def __init__(self,llm,prompt):
        self.llm=llm
        self.prompt=prompt

    def run(self,input_dict):
        final_prompt=self.prompt.format(input_dict)
        result=self.llm.predict(final_prompt)
        return result['response']
    
template=fakePromptTemplate(
    template='write a poem about {topic}',
    input_variables={'topic'}
)   

llm=fakeLLM()
chain=fakeLLMChain(llm,template)
result=chain.run({'topic':'India'})
print(result)