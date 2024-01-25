import square2

print(square2.base)
print(square2.square(10))

from square2 import base, square
print(base)

square(10)

#############################################
import person
maria = person.Person('마리아', 20, '수원시')
maria.greeting()

##############################################
from person import Person
maria = Person('마리아', 20, '수원시')
maria.greeting()

##############################################
import calc
calc.add(50, 60)
calc.mul(50, 60)
