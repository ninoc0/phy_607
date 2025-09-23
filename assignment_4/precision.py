#%%
class Precision():
    def __init__(self, num):
        self.num = "{:.9f}".format(num)
        #return num
    def printing(self):
        print(self.num)
# %%
a = Precision(7.1)
a.printing()
# %%
