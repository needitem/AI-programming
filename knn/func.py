import numpy as np
import matplotlib.pyplot as plt
import pickle
import pdb
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def init_data2():
    with open("../Data/apple_quality.csv") as f1:
        apple = pd.read_csv(f1)
    return apple


def split_train_test(data, test_ratio):
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(data))
    # drop last column and convert to numpy array
    x = data.drop("Quality", axis=1).values
    y = data["Quality"].values

    test_set_size = int(len(data) * test_ratio)
    x_train = x[test_set_size:]
    y_train = y[test_set_size:]
    x_test = x[:test_set_size]
    y_test = y[:test_set_size]

    return x_train, y_train, x_test, y_test


def one_hot_encode(data):
    encoder = OneHotEncoder()
    encoder.fit(data)
    return encoder.transform(data).toarray()

def knn2(x_train, y_train, x_test, k):
    n = len(x_test)
    y_pred = np.zeros(n)
    for i in range(n):
        dist = np.sum((x_train - x_test[i]) ** 2, axis=1)
        idx = np.argsort(dist)[:k]
        y_pred[i] = np.argmax(np.bincount(y_train[idx]))
    return y_pred



def calculateMeasure2(result, y_test):
    TP, TN, FP, FN = [], [], [], []
    for i in range(2):
        TP.append(((result == y_test) & (y_test == i)).sum())
        TN.append(((result != i) & (y_test != i)).sum())
        FP.append(((result != y_test) & (y_test == i)).sum())
        FN.append(((result == i) & (y_test != i)).sum())

    TP = np.array(TP)
    TN = np.array(TN)
    FP = np.array(FP)
    FN = np.array(FN)

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)

    return accuracy, precision, recall, f1


def pca(trainSet, testSet, k):
    # tmp = np.cov(trainSet.T)
    # L, V = np.linalg.eig(tmp)
    U, s, V = np.linalg.svd(
        trainSet.T
    )  # U: eigenvectors, s: eigenvalues, V: eigenvectors

    plt.plot(s)  # explained variance
    plt.xlabel("Number of Components")
    plt.ylabel("Explained Variance")
    plt.show()
    U = U[:, np.argsort(s)[::-1]]
    PCA = U[:, 0:k]
    trainSet = np.dot(trainSet, PCA)
    testSet = np.dot(testSet, PCA)

    return trainSet, testSet
