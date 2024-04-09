import numpy as np
import matplotlib.pyplot as plt
import pickle
import pdb
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


PATH = "./4주MNIST/"


def init_data():
    with open(PATH + "train.bin", "rb") as f1:
        train = pickle.load(f1)
    with open(PATH + "test.bin", "rb") as f2:
        test = pickle.load(f2)

    return train, test


def init_data2():
    with open("../apple_quality.csv") as f1:
        apple = pd.read_csv(f1)
    return apple


def prepareData(train, test, k=300):
    trainSet = []
    testSet = []

    for i in range(10):
        trainSet.append(train[i][0:k])
        testSet.append(test[i][0:100])
    return trainSet, testSet


def prepareData2(train, test, k=300):
    trainSetf = np.zeros((k * 10, 28 * 28))
    testSetf = np.zeros((100 * 10, 28 * 28))

    for i in range(len(train)):
        for j in range(k):
            trainSetf[i * k + j] = train[i][j].flatten()
    for i in range(len(test)):
        for j in range(100):
            testSetf[i * 100 + j] = test[i][j].flatten()

    return trainSetf, testSetf


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


def knn(trainSet, testSet, k):
    trainSet1, trainsSet2 = trainSet.shape
    testSet1, testSet2 = testSet.shape
    trainSet3 = int(trainSet1 / 10)
    testSet3 = int(testSet1 / 10)

    label = np.tile(np.arange(0, 10), (testSet3, 1))  # answer sheet
    result = np.zeros((testSet3, 10))

    for i in range(testSet1):
        tmp = np.sum((trainSet - testSet[i, :]) ** 2, axis=1)
        no = np.argsort(tmp)[0:k]
        hist, bins = np.histogram(no // trainSet3, np.arange(-0.5, 10.5, 1))
        result[i % testSet3, i // testSet3] = np.argmax(hist)
    return result


def knn2(x_train, y_train, x_test, k):
    n = len(x_test)
    y_pred = np.zeros(n)
    for i in range(n):
        dist = np.sum((x_train - x_test[i]) ** 2, axis=1)
        idx = np.argsort(dist)[:k]
        y_pred[i] = np.argmax(np.bincount(y_train[idx]))
    return y_pred


def createTemplate(trainSet):
    template = np.zeros((28, 28 * 10))
    for i in range(10):
        tmp = np.array(trainSet[i])
        template[:, i * 28 : (i + 1) * 28] = np.mean(tmp, axis=0)
    return template


def matchTemplate(template, testSet):
    result = np.zeros((100, 10))
    for i in range(len(testSet)):
        for j in range(len(testSet[0])):
            tmp = np.tile(testSet[i][j], (1, 10))
            error = np.abs(template - tmp)
            errorSum = [np.sum(error[:, j : j + 28]) for j in range(0, 280, 28)]
            result[j, i] = np.argmin(errorSum)
    return result


def calculateMeasure(result):
    row, col = result.shape
    label = np.tile(np.arange(0, col), (row, 1))
    # print(label, result)
    TP, TN, FP, FN = [], [], [], []
    for i in range(10):
        TP.append(((result == label) & (label == i)).sum())
        TN.append(((result != i) & (label != i)).sum())
        FP.append(((result != label) & (label == i)).sum())
        FN.append(((result == i) & (label != i)).sum())

    TP = np.array(TP)
    TN = np.array(TN)
    FP = np.array(FP)
    FN = np.array(FN)

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)

    return accuracy, precision, recall, f1


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


def feature1(trainSet, testSet):
    trainSet1 = len(trainSet)
    trainSet2 = len(trainSet[0])
    testSet1 = len(testSet)
    testSet2 = len(testSet[0])

    trainSetf = np.zeros((trainSet1 * trainSet2, 5))
    testSetf = np.zeros((testSet1 * testSet2, 5))

    for i in range(trainSet1):
        for j in range(trainSet2):
            tmp = trainSet[i][j]
            tmp = np.where(tmp != 0)
            tmp2 = np.mean(tmp, 1)
            tmp3 = np.cov(tmp)
            trainSetf[i * trainSet2 + j, :] = np.array(
                [tmp2[0], tmp2[1], tmp3[0, 0], tmp3[0, 1], tmp3[1, 1]]
            )

    for i in range(testSet1):
        for j in range(testSet2):
            tmp = testSet[i][j]
            tmp = np.where(tmp != 0)
            tmp2 = np.mean(tmp, 1)
            tmp3 = np.cov(tmp)
            testSetf[i * testSet2 + j, :] = np.array(
                [tmp2[0], tmp2[1], tmp3[0, 0], tmp3[0, 1], tmp3[1, 1]]
            )

    return trainSetf, testSetf


def feature2(trainSet, testSet, dx):
    size = trainSet[0][0].shape[0]
    s = size - dx + 1

    trainSet1 = len(trainSet)
    trainSet2 = len(trainSet[0])
    testSet1 = len(testSet)
    testSet2 = len(testSet[0])

    traintmp = np.zeros((trainSet1 * trainSet2, s, s))
    testtmp = np.zeros((testSet1 * testSet2, s, s))
    trainSetf = np.zeros((trainSet1 * trainSet2, s * s))
    testSetf = np.zeros((testSet1 * testSet2, s * s))

    for i in range(trainSet1):
        for j in range(trainSet2):
            tmp = trainSet[i][j]
            for k in range(s):
                for l in range(s):
                    traintmp[i * trainSet2 + j, k, l] = tmp[
                        k : k + dx, l : l + dx
                    ].sum()
            trainSetf[i * trainSet2 + j, :] = traintmp[i * trainSet2 + j, ::].flatten()

    for i in range(testSet1):
        for j in range(testSet2):
            tmp = testSet[i][j]
            for k in range(s):
                for l in range(s):
                    testtmp[i * testSet2 + j, k, l] = tmp[k : k + dx, l : l + dx].sum()
            testSetf[i * testSet2 + j, :] = testtmp[i * testSet2 + j, ::].flatten()

    return trainSetf, testSetf


def feature3(trainSet, testSet):
    # replace nan to mean
    trainSet = np.nan_to_num(trainSet)
    testSet = np.nan_to_num(testSet)
    return trainSet, testSet


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
