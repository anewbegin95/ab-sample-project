# %%
import pandas as pd
import numpy as np
import math

baseline_csv = 'baseline_clickthrough_rate.csv'

df = pd.read_csv(baseline_csv)
# print(df.head())

x_bar = df['Clickthrough_Rate'].mean()
x_vals = df['Clickthrough_Rate'].tolist()
k = len(x_vals)

# print(x_bar)
# print(x_vals)
# print(k)

sum = 0

for x_i in x_vals:
    sum += (x_i - x_bar) ** 2

var = (sum / (k - 1)) * 2
print(var)
# %%
