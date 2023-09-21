# %%
import pandas as pd
import numpy as np
import statsmodels.stats.power as smp
from scipy.stats import norm

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
submissions_1 = 200
submissions_2 = 235
p1 = submissions_1 / sample_size
p2 = submissions_1 / sample_size
pooled_p = (p1 * sample_size + p2 * sample_size)/sample_size + sample_size
se = np.sqrt((pooled_p * (1 - pooled_p)) * (1/sample_size + 1/sample_size))
z = (p1 - p2) / se
alpha = 0.05
critical_val = norm.ppf(1 - alpha/2)
print(z, critical_val)
# %%
