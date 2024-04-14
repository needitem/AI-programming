import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

apple = pd.read_csv("./Data/apple_quality.csv")
apple["Quality"] = apple["Quality"].apply(lambda x: 1 if x == "good" else 0)
columns = apple.columns

scaler = StandardScaler()

x = apple.drop(columns=["Quality"]).values
y = apple["Quality"].values

x = scaler.fit_transform(x)

pca = PCA(n_components=len(x[0]))
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(
    data=principalComponents, columns=[f"PC{i}" for i in range(1, len(x[0]) + 1)]
)

finalDf = pd.concat([principalDf, apple[["Quality"]]], axis=1)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel("Principal Component 1", fontsize=15)
ax.set_ylabel("Principal Component 2", fontsize=15)
ax.set_title("2 component PCA", fontsize=20)

# draw the eigenVectors, the direction of the most variance. The longer the vector, the more variance. draw lines from the mean to the eigenVectors
for i, v in enumerate(pca.components_):
    ax.plot(
        [0, v[0] * 3],
        [0, v[1] * 3],
        color="black",
        lw=2,
    )
    ax.text(v[0] * 3, v[1] * 3, columns[i], ha="right")

targets = [0, 1]
colors = ["r", "g"]
for target, color in zip(targets, colors):
    indicesToKeep = finalDf["Quality"] == target
    ax.scatter(
        finalDf.loc[indicesToKeep, "PC1"],
        finalDf.loc[indicesToKeep, "PC2"],
        c=color,
        s=50,
    )
ax.legend(targets)
ax.grid()
plt.show()
