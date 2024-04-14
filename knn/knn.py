import func as fs

apple = fs.init_data2()
apple = apple.drop(columns=["A_id"])
apple["Quality"] = apple["Quality"].apply(lambda x: 1 if x == "good" else 0)
apple = apple.dropna(inplace=False)

# x:data y:label
x_train, y_train, x_test, y_test = fs.split_train_test(
    apple, 0.2
)  # x for data, y for label, 0.2 for test size

x_train, x_test = fs.pca(x_train, x_test, k=len(x_train[0]))  # k : amount of components
result = fs.knn2(x_train, y_train, x_test, k=6)  # k : amount of neighbors
acc, pre, rec, f1 = fs.calculateMeasure2(result, y_test)

print("Accuracy: ", acc)
print("Precision: ", pre)
print("Recall: ", rec)
print("F1: ", f1)
