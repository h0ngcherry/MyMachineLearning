class Myself:
    def __init__(self, name, age):
        self.name = name
        self.age = age
    def cht(self):
        print(f'hi!')
    def cht(self, name, age):
        print(f'hi {name} {age}')
try:
    cht = Myself()
except TypeError:
    print("不能调用无参数的构造函数")

cht = Myself("cht", 18)
cht.cht("new_name", 20)  