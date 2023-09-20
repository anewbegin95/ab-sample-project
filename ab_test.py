# %%
import pandas as pd
import numpy as np
import statsmodels.stats.power as smp

# %%
baseline_csv = 'baseline_clickthrough_rate.csv'

df = pd.read_csv(baseline_csv)
print(df.head())

# %%
x_vals = [x * 100 for x in df['Clickthrough_Rate'].tolist()]
x_bar = np.mean(x_vals)
k = len(x_vals)

print(x_bar)
print(x_vals)
print(k)

sum = 0

for x_i in x_vals:
    sum += (x_i - x_bar) ** 2

var = (sum / (k - 1)) * 2
print(var)
# %%
delta_squared = (.05 - (x_bar / 100)) ** 2
print(delta_squared)
# %%
z_crit_alpha = 1.96
z_crit_beta = .84

sample_size = (((z_crit_alpha + z_crit_beta) ** 2) * var) / delta_squared
two_sample_size = sample_size * 2
print(sample_size, two_sample_size)
# %%
