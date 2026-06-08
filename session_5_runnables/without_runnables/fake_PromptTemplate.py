class fakePromptTemplate:
    def __init__(self,template,input_variables):
        self.template=template
        self.input_variables=input_variables

    def format(self,input_dict):
        return self.template.format(**input_dict)
    
template=fakePromptTemplate(
    template='write a poem about {topic}',
    input_variables={'topic'}
)    

result=template.format({'topic':'india'})
print(result)