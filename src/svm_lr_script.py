"""
Takes the clean data location get the data from their do the model evaluation and train
it using SVM and LR and export the reports and results of SVM, LR model to the export
location

Usage: src/svm_lr_script.py data_loc export_loc

Options:
data_loc     The location of the cleaned data that the model will use to train
export_loc   The export location of all the results based on the KNN model
"""

import pandas as pd
import argparse
from line_plot import *
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np
import matplotlib.pyplot as plt
from train_and_predict_model import *
from para_optimize import *
from std_acc import *

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

# now in next steps let's do the KNN training
y = zoo_data['type']

# splitting the dataset 80-20 for train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

# final LR model is here used the splited test part to train again for better training, and better prediction
# LR evaluation train and final model is also here
LR = finalModel("LR", -1, X_train, X_test, y_train, y_test, X, y)
yhat = LR.predict(X_test)

report = classification_report(y_test, yhat, output_dict=True)
df = pd.DataFrame(report).transpose()
df.to_csv(export_loc + "csv/lr_classification_report.csv")

# Final SVM is here used the splited test part to train again for better training, and better prediction
# svm evaluation train and final model is also here scroll through the output
svec = finalModel("SVM", -1, X_train, X_test, y_train, y_test, X, y)
yhat = svec.predict(X_test)

report = classification_report(y_test, yhat, output_dict=True)
df = pd.DataFrame(report).transpose()
df.to_csv(export_loc + "csv/svm_classification_report.csv")
