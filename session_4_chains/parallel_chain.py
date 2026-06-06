from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

llm1 = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task='text-generation'
)

model1=ChatHuggingFace(llm=llm1)

llm2 = HuggingFaceEndpoint(
    repo_id="katanemo/Arch-Router-1.5B",
    task='text-generation'
)

model2=ChatHuggingFace(llm=llm2)

prompt1=PromptTemplate(
    template='Generate short and simple notes from the following text \n {text}',
    input_variables=['text']
)

prompt2=PromptTemplate(
    template='Generate 5 short question answers from the following text\n {text}',
    input_variables=['text']
)

prompt3=PromptTemplate(
    template='Merge the provided notes and quiz into a single document \n notes -> {notes} and quiz -> {quiz}',
    input_variables=['notes','quiz']
)

parser=StrOutputParser()

parallel_chain=RunnableParallel({
    'notes':prompt1|model1|parser,
    'quiz':prompt2|model2|parser
})

merge_chain=prompt3|model1|parser

chain=parallel_chain|merge_chain

text="""
The laws of thermodynamics are a set of scientific laws which define a group of physical quantities, such as temperature, energy, and entropy, that characterize thermodynamic systems in thermodynamic equilibrium. The laws also use various parameters for thermodynamic processes, such as thermodynamic work and heat, and establish relationships between them. They state empirical facts that form a basis of precluding the possibility of certain phenomena, such as perpetual motion. In addition to their use in thermodynamics, they are important fundamental laws of physics in general and are applicable in other natural sciences.

Traditionally, thermodynamics has recognized three fundamental laws, simply named by an ordinal identification, the first law, the second law, and the third law.[1][2][3] A more fundamental statement was later labelled as the zeroth law after the first three laws had been established.

The zeroth law of thermodynamics defines thermal equilibrium and forms a basis for the definition of temperature: if two systems are each in thermal equilibrium with a third system, then they are in thermal equilibrium with each other.

The first law of thermodynamics states that, when energy passes into or out of a system (as work, heat, or matter), the system's internal energy changes in accordance with the law of conservation of energy. This also results in the observation that, in an externally isolated system, even with internal changes, the sum of all forms of energy must remain constant, as energy cannot be created or destroyed.

The second law of thermodynamics states that in a natural thermodynamic process, the sum of the entropies of the interacting thermodynamic systems never decreases. A common corollary of the statement is that heat does not spontaneously pass from a colder body to a warmer body.

The third law of thermodynamics states that a system's entropy approaches a constant value as the temperature approaches absolute zero. With the exception of non-crystalline solids (glasses), the entropy of a system at absolute zero is typically close to zero.[2]

The first and second laws prohibit two kinds of perpetual motion machines, respectively: the perpetual motion machine of the first kind which produces work with no energy input, and the perpetual motion machine of the second kind which spontaneously converts thermal energy into mechanical work. 
"""

result=chain.invoke({'text':text})

print(result)
