"""
Takes the clean data location get the data from their do the model evaluation and train
it using the best k for KNN and export the reports and results of KNN model to the export
location

Usage: src/knn_script.py data_loc export_loc

Options:
data_loc     The location of the cleaned data that the model will use to train
export_loc   The export location of all the results based on the KNN model
"""

import pandas as pd
import argparse
from src.zoo.line_plot import *
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np
import matplotlib.pyplot as plt
from src.zoo.train_and_predict_model import *
from src.zoo.para_optimize import *
from src.zoo.std_acc import *

# setting up the parser
parser = argparse.ArgumentParser(description='Read and save dataset')
parser.add_argument('data_loc', type=str, help='The Location of Cleaned Data')
parser.add_argument('export_loc', type=str, help='The Location for exporting Figures and Data')
args = parser.parse_args()

# setting up the variables
data_loc = args.data_loc
export_loc = args.export_loc

# reading the data
zoo_data = pd.read_csv(data_loc)

# extracting the feature
feature = zoo_data[["hair", "feathers", "eggs", "milk", "airborne",
                   "aquatic", "predator", "toothed", "backbone", "breathes",
                   "venomous", "fins", "legs", "tail", "domestic", "catsize"]]

# making it as X
X = feature

# saving the top 5 row of data as csv
X.head().to_csv(export_loc + "csv/head.csv")

# now in next steps let's do the KNN training
y = zoo_data['type']

# splitting the dataset 80-20 for train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

# training the model for different set of K values and finding the best K value
Ks = 81
mean_acc = np.zeros((Ks - 1))
std_acc = np.zeros((Ks - 1))
ConfustionMx = [];
for n in range(1, Ks):
    # Train Model and Predict
    neigh = KNeighborsClassifier(n_neighbors=n).fit(X_train, y_train)
    yhat = neigh.predict(X_test)
    mean_acc[n - 1] = metrics.accuracy_score(y_test, yhat)

std_acc = std_acc(yhat, y_test, Ks)

line_plot(Ks, mean_acc, std_acc, "Number of Neighbors (K)", "Accuracy", "Number of Neighbors vs. Accuracy")
plt.savefig(export_loc + "figures/k_accuracy.png")

# Finding the K value using Grid Search
knn = KNeighborsClassifier()
k_vals = list(range(1, 21))
param_grid = dict(n_neighbors=k_vals)

para_optimize(knn, param_grid, 3, X_train, y_train)

# as the best accuracy was with K = 1
# using K = 1 for the final KNN model
# Final KNN model is here used the split test part to train again for better training, and better prediction
# KNN evaluation is also here scroll through the output
final_knn_model = finalModel("KNN", 1, X_train, X_test, y_train, y_test, X, y)

# cross-validation on knn
cv_results_knn = cross_validate(final_knn_model, X_train, y_train, cv=3, return_train_score=True);
pd.DataFrame(cv_results_knn).mean().to_csv(export_loc + "csv/knn_cross_validate_result.csv")
yhat = final_knn_model.predict(X_test)
report = classification_report(y_test, yhat, output_dict=True)
df = pd.DataFrame(report).transpose()
df.to_csv(export_loc + "csv/knn_classification_report.csv")
