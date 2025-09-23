#%%
class Precision():
    def __init__(self, num):
        self.num = "{:.9f}".format(num)
        #return num
    def printing(self):
        print(self.num)
    def __add__(self, a, b):
        return self.a + self.b
    def __sub__(self, a, b):
        return self.a - self.b
    def __mul__(self, a, b):
        return self.a * self.b
    def __div__(self, a, b):
        return self.a / self.b
# %%
a = Precision(7.1)
a.printing()
# %%
