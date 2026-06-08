import random

class fakeLLM:
    def __init__(self):
        print('LLM created')

    def predict(self,prompt):

        response_list=[
            'Delhi is the capital of India',
            'IPL is a cricket league',
            'AI stands for Artificial Intelligence'
        ]    

        return {'response': random.choice(response_list)}
    
llm=fakeLLM()
result=llm.predict('What is the capital of India')    
print(result)