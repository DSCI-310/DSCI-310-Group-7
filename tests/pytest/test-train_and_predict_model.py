import unittest
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split

from src.train_and_predict_model import *

dataset_iris = sm.datasets.get_rdataset(dataname='iris', package='datasets')
df_iris = dataset_iris.data
feature = df_iris[["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"]]
X = feature
y = df_iris["Species"]
k_1 = 10
k_2 = 9
k_3 = 8
# function input for the test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
msg = "expected value and exact output is not equal!"


class TestTrainAndPredictModel(unittest.TestCase):
    def test_df_case(self):
        algorithm = "DT"
        mean_acc = np.zeros((k_1 - 1))
        self.assertEqual(9, len(train_and_predict_model(algorithm, k_1, X_train, X_test, y_train, y_test, mean_acc)),
                         msg)
        mean_acc = np.zeros((k_2 - 1))
        self.assertEqual(8, len(train_and_predict_model(algorithm, k_2, X_train, X_test, y_train, y_test, mean_acc)),
                                msg)
        mean_acc = np.zeros((k_3 - 1))
        self.assertEqual(7, len(train_and_predict_model(algorithm, k_3, X_train, X_test, y_train, y_test, mean_acc)),
                                 msg)

    def test_knn_case(self):
        algorithm = "KNN"
        mean_acc = np.zeros((k_1 - 1))
        self.assertEqual(9, len(train_and_predict_model(algorithm, k_1, X_train, X_test, y_train, y_test, mean_acc)),
                         msg)
        mean_acc = np.zeros((k_2 - 1))
        self.assertEqual(8, len(train_and_predict_model(algorithm, k_2, X_train, X_test, y_train, y_test, mean_acc)),
                                msg)
        mean_acc = np.zeros((k_3 - 1))
        self.assertEqual(7, len(train_and_predict_model(algorithm, k_3, X_train, X_test, y_train, y_test, mean_acc)),
                                 msg)
