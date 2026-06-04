from pydantic import BaseModel, EmailStr, Field

class Student(BaseModel):
    name: str = 'soumil'             # default value = 'soumil'
    email: EmailStr
    cgpa: float = Field(gt=0, lt=10)

# new_student = {'name':'soumil'}    (if default value is not set)

# new_student = {'email':'abc'}        # Not a valid email address

new_student = {'email':'abc@gmail.com', 'cgpa':8}

student=Student(**new_student)

# print(student.name)               # fetch attributes

student_dict = student.model_dump_json()

print(student_dict)