from langchain_text_splitters import RecursiveCharacterTextSplitter,Language

text="""

def check_even_odd(num):
    if num % 2 == 0:
        return "Even"
    else:
        return "Odd"


number = int(input("Enter a number: "))

result = check_even_odd(number)

print(f"{number} is {result}")

"""

splitter=RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=500,
    chunk_overlap=0,
)

chunks=splitter.split_text(text)

print(len(chunks))
print(chunks[0])