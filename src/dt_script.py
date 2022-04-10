"""
Takes the clean data location get the data from their do the model evaluation and train
it using the best max-depth for decision tree and export the reports and results of decision tree model to the export
location

Usage: src/dt_script.py data_loc export_loc

Options:
data_loc     The location of the cleaned data that the model will use to train
export_loc   The export location of all the results based on the KNN model
"""

import pandas as pd
import argparse
# from line_plot import *
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np
import matplotlib.pyplot as plt
from src.zoo import train_and_predict_model, para_optimize, std_acc, line_plot
from sklearn import metrics
from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import jaccard_score, classification_report
# from train_and_predict_model import *
# from para_optimize import *
# from std_acc import *

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

# now in next steps let's do the Decision Tree modeling
y = zoo_data['type']

# splitting the dataset 80-20 for train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

Ks = 50
mean_acc = np.zeros((Ks-1))
sd_acc = np.zeros((Ks-1))
ConfustionMx = [];
for n in range(1,Ks):
    # Train Model and Predict
    decTree = DecisionTreeClassifier(criterion="entropy", max_depth = n)
    decTree.fit(X_train,y_train)
    yhat=decTree.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)

sd_acc = std_acc.std_acc(yhat, y_test, Ks)

line_plot.line_plot(Ks, mean_acc, sd_acc, "Max Depth", "Accuracy", "Max Depth vs. Accuracy")
plt.savefig(export_loc + "figures/dt_accuracy.png")

# As Best is max depth = 5
# using max depth = 5 for the final decision tree
# Final decision tree is here used the split test part to train again for better training, and better prediction
# DT evaluation is also here scroll through the output
Final_dec_Tree = train_and_predict_model.final_Model("DT", 5, X_train, X_test, y_train, y_test, X, y)

# cross-validation on decision tree
cv_results_dt = cross_validate(Final_dec_Tree, X_train, y_train, cv=4, return_train_score=True);
pd.DataFrame(cv_results_dt).mean().to_csv(export_loc + "csv/dt_cross_validate_result.csv")
yhat = Final_dec_Tree.predict(X_test)
report = classification_report(y_test, yhat, output_dict=True)
df = pd.DataFrame(report).transpose()
df.to_csv(export_loc + "csv/dt_classification_report.csv")
