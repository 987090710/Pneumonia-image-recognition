# -- coding: utf-8 --
import pandas as pd
import numpy as np
pd.options.display.max_columns=999
orders = pd.read_excel("C:/Users/Zz/Desktop/YH/网单记录.xls")

group = orders.groupby(["货号"])
s = group["数量"].sum()
df = pd.DataFrame({"数量":s})
print(df)
df.to_excel("C:/Users/Zz/Desktop/YH/大货总计.xls")
s = group["样品"].sum()
df = pd.DataFrame({"数量":s})
print(df)
df.to_excel("C:/Users/Zz/Desktop/YH/样本总计.xls")
s = group["鞋垫"].sum()
df = pd.DataFrame({"数量":s})
print(df)
df.to_excel("C:/Users/Zz/Desktop/YH/鞋垫总计.xls")