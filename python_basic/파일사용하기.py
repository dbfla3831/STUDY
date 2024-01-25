# 파일에 문자열 쓰기
# file = open('hello.txt', 'w')
# file.write('hello, world!')
# file.close()

# # 파일에서 문자열 읽기
# file = open('hello.txt', 'r')
# s = file.read()
# print(s)
# file.close()

# 자동으로 파일 객체 닫기
# with open('hello.txt', 'r') as f:
#     s = f.read()
#     print(s)

# 문자열 여러 줄을 파일에 쓰기, 읽기
# with open('hello.txt', 'w') as file:
#     for i in range(3):
#         file.write('hello, world!{}\n'.format(i))

# 리스트에 들어있는 문자열을 파일에 쓰기
# lst = ['cat\n', 'dog\n', 'rabbit\n']
# with open("hello.txt", 'w') as file:
#     file.writelines(lst)

# # 파일의 내용을 한 줄씩 리스트로 가져오기
# with open("hello.txt", 'r') as file:
#     lst2 = file.readlines()
#     print(lst2)

# # 파일의 내용을 한 줄씩 읽기
# with open('hello.txt', 'r') as file:
#     line = None
#     while line != '':
#         line = file.readline()
#         print(line.strip('\n'))

# # for 반복문으로 파일의 내용을 줄 단위로 읽기
# with open('hello.txt', 'r') as file:
#     for line in file:
#         print(line.strip('\n'))


# 파이썬 객체를 파일에 저장하기
# import pickle

# name = "james"
# age = 17
# address = "서울시 서초구 반포동"
# scores = {'kor' : 90, 'eng' : 95, 'mat' : 85, 'sci' : 82}

# with open('james.p', 'wb') as file:
#     pickle.dump(name, file)
#     pickle.dump(age, file)
#     pickle.dump(address, file)
#     pickle.dump(scores, file)

# # 파일에서 파이썬 객체 읽기
# import pickle
# with open('james.p', 'rb') as file:
#     name = pickle.load(file)
#     age = pickle.load(file)
#     address = pickle.load(file)
#     scores = pickle.load(file)
#     print(name)
#     print(age)
#     print(address)
#     print(scores)


# 연습
# while True:
#     name = input("동물 이름 입력 : ")
#     if name != '0000':
#         with open('animals.txt', 'a') as file:
#             file.write(name + '\n')
#     else:
#         break

# 연습
# 27.5 연습 문제

# lst = ['anonymously\n', 'compatibility\n', 'dashboard\n', 'experience\n',
#        'photography\n', 'spotlight\n', 'warehouse\n']
# with open('words.txt', 'w') as file:
#     file.writelines(lst)

# with open('words.txt', 'r') as file:
#     count = 0
#     words = file.readlines()
#     for word in words:
#         if len(word.strip('\n')) <= 10:
#             count += 1
# print(count)

# 27.6
# lst = ['Fortunately', 'however', 'for the reputation of Asteroid B-612',
#        'a Turkish dictator made a law that his subjects', 'under pain of death', 
#        'should change to European costume. So in 1920 the astronomer gave his demonstration all over again',
#        'dressed with impressive style and elegance. And this time everybody accepted his report.']
# with open('words.txt', 'w') as file:
#     for i in range(len(lst)):
#         if i != len(lst)-1:
#             file.write(lst[i] + ', ')
#         else:
#             file.write(lst[i])

# with open('words.txt', 'r') as file:
#     words = file.readline()
#     for word in words.split():
#         if 'c' in word:
#             print(word.strip(',.'))
