import pandas as pd
import func as fs
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import time
import pdb

apple = fs.init_data2()
apple = apple.drop(columns=["Weight", "A_id", "Weight", "Crunchiness", "Acidity"])
apple["Quality"] = apple["Quality"].apply(lambda x: 1 if x == "good" else 0)
apple = apple.dropna(inplace=False)
# x:data y:label
x_train, y_train, x_test, y_test = fs.split_train_test(apple, 0.2)
x_train, x_test = fs.pca(x_train, x_test, 4)
result = fs.knn2(x_train, y_train, x_test, 6)
acc, pre, rec, f1 = fs.calculateMeasure2(result, y_test)

print("Accuracy: ", acc)
print("Precision: ", pre)
print("Recall: ", rec)
print("F1: ", f1)
