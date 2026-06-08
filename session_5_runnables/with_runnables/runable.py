from abc import ABC, abstractmethod
import random
class Runnable(ABC):
    @abstractmethod
    def invoke(input_data):
        pass


class fakeLLM(Runnable):
    def __init__(self):
        print('LLM created')

    def invoke(self,prompt):

        response_list=[
            'Delhi is the capital of India',
            'IPL is a cricket league',
            'AI stands for Artificial Intelligence'
        ]    

        return {'response': random.choice(response_list)}        

    def predict(self,prompt):

        response_list=[
            'Delhi is the capital of India',
            'IPL is a cricket league',
            'AI stands for Artificial Intelligence'
        ]    

        return {'response': random.choice(response_list)}    
    

class fakePromptTemplate:
    def __init__(self,template,input_variables):
        self.template=template
        self.input_variables=input_variables

    def invoke(self,input_dict):
        return self.template.format(**input_dict)    

    def format(self,input_dict):
        return self.template.format(**input_dict)
    

class RunnableConnector(Runnable):
    def __init__(self,runnable_list):
        self.runnable_list=runnable_list

    def invoke(self,input_data):
        for runnable in self.runnable_list:
            input_data=runnable.invoke(input_data)    

        return input_data


class fakeStringOutput(Runnable):
    def __init__(self):
        pass

    def invoke(self, input_data):
        return input_data['response']
       

template=fakePromptTemplate(
    template='write a poem about {topic}',
    input_variables={'topic'}
) 

llm=fakeLLM()

parser=fakeStringOutput()

# chain=RunnableConnector([template,llm,parser])

# result=chain.invoke({'topic':'india'})

# print(result)

template1=fakePromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

template2=fakePromptTemplate(
    template='Explain the following joke {response}',
    input_variables=['response']
)

chain=RunnableConnector([template1,llm])

# response=chain.invoke({'topic':'AI'})

chain2=RunnableConnector([template2,llm,parser])

# response1=chain2.invoke({'response':response})

final_chain=RunnableConnector([chain,chain2])

response=final_chain.invoke({'topic':'cricket'})

print(response)


