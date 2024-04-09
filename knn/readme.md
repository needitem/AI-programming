# Apple Quality Classification using KNN and PCA

This project demonstrates the classification of apple quality using the K-Nearest Neighbors (KNN) algorithm and Principal Component Analysis (PCA) on the "apple_quality.csv" dataset.

## Dataset

The dataset used in this project contains information about various attributes of apples, including weight, crunchiness, acidity, and quality. The "Quality" column is the target variable, which classifies the apples as either "good" or "bad".

## Prerequisites

- Python 3.x
- numpy
- matplotlib
- pickle
- pandas
- scikit-learn

## Usage

1. Ensure you have the necessary libraries installed.
2. Place the "apple_quality.csv" file in the "../Data/" directory.
3. Run the provided code, which will perform the following steps:
   - Load the data using the `init_data2()` function
   - Split the data into training and testing sets using the `split_train_test()` function
   - Apply PCA to the training and testing sets using the `pca()` function
   - Perform KNN classification using the `knn2()` function
   - Calculate the accuracy, precision, recall, and F1-score using the `calculateMeasure2()` function
4. The results of the classification will be printed to the console.

## Functions

1. `init_data2()`: Loads the "apple_quality.csv" file and returns the data as a pandas DataFrame.
2. `split_train_test(data, test_ratio)`: Splits the data into training and testing sets based on the given test ratio.
3. `one_hot_encode(data)`: Performs one-hot encoding on the input data.
4. `knn2(x_train, y_train, x_test, k)`: Implements the KNN algorithm to classify the test data.
5. `calculateMeasure2(result, y_test)`: Calculates the accuracy, precision, recall, and F1-score of the classification results.
6. `pca(trainSet, testSet, k)`: Applies Principal Component Analysis to the training and testing sets, reducing the number of features to `k`.

## Results

The code will print the following metrics after running the classification:

- Accuracy
- Precision
- Recall
- F1-score

Feel free to explore the code and modify it to experiment with different parameters or techniques.
