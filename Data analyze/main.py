import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

apple = pd.read_csv("./Data/apple_quality.csv")
print(apple.head())
apple["Quality"] = apple["Quality"].apply(lambda x: 1 if x == "good" else 0)

info = apple.info()
print(info)
describe = apple.describe()
print(describe)
apple.hist(bins=50, figsize=(20, 15))
plt.show()

corr_matrix = apple.corr()
corr_matrix["Quality"].sort_values(ascending=False)
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.show()
