#!/usr/bin/env python
# coding: utf-8

# # Methods & Results
# We are going to use multiple analysis to classify the type of the animals using 16 variables including hair, feathers, eggs, milk, airborne, aquatic, predator, toothed, backbone, breathes, venomous, fins, legs, tail, domestic, catsize as our predictors. To predict the class of a new observation, the algorithms of each type will be further explained before implementation.

# In[1]:


{
    "tags": [
        "hide-cell"
    ]
}
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append( '..' )
from src.pre_processing import *
from src.train_and_predict_model import *
from src.line_plot import *
from src.para_optimize import *
from src.std_acc import *
from src.line_plot import *


# The first thing is to import the data. The data set is downloaded from [UCI repository]("https://archive-beta.ics.uci.edu/ml/datasets/zoo"). It is then saved as a csv file in this project repository. Some exploratory data analysis needs to be run before running the actual analyses on the data set. Here is a preview of pre-processed data set:

# In[2]:


{
    "tags": [
        "hide-input"
    ]
}
zoo_data = pd.read_csv("../results/csv/head.csv")
zoo_data.columns.values[0] = "index"
zoo_data


# It is checked that there aren't missing values in the data set, we can clearly deduce that the data set is clean according to the data summary we generated above. Since most features are binary and categorical, there is no need to do normalization and standardization.

# ```{figure-md} f1
# <img src="../results/figures/fig1.png" alt="num" class="bg-primary mb-1" width="800px">
# 
# A summary table of the data set
# ```

# As shown in [fig.1](f1), the histograms of each feature are generated. The ones with skewed distribution might be more decisive in the prediction. However, since the data set is relatively small, all the features except the `animalName` are going to be used to predict. In the next part, we are going to split the data, into the training set and testing set. After that, different classification models will be trained and evaluated.

# ## Classification
# Now we will use the training set to build an accurate model, whereas the testing set is used to report the accuracy of the models. Here is a list of algorithms we will use in the following section:
# 
# - K Nearest Neighbor(KNN)
# - Decision Tree
# - Support Vector Machine
# - Logistic Regression

# To train and evaluate each model, we split the dataset into training and testing sets. We use 80% of the total data to train the models, and the rest of the data is aimed to test the models.

# ### KNN
# KNN captures the idea of similarity (sometimes called distance, proximity, or closeness) 
# with some basic mathematics we might have learned earlier. Basically in terms of geometry we can always calculate the distance between points on a graph. Similarly, using KNN we can group similar points together and predict the target with our feature variables(x).

# ```{figure-md} f2
# <img src="../results/figures/k_accuracy.png" alt="num" class="bg-primary mb-1" width="500px">
# 
# A plot reveals the relationship between K and corresponding accuracy
# ```

# As shown in [fig.2](f2), less K values provide higher accuracy. To find the best K value, we tuned the hyperparameter using `GridSearch` algorithm. After tuning, the best K value is 1.

# ### KNN final model & Evaluation

# After fitting the model using `K=1`, we evaluate the KNN model by Cross Validation and calculating the precision, recall, f1-score and support.

# KNN Cross Validation Result:

# In[3]:


{
    "tags": [
        "hide-input"
    ]
}
knn_cross_validate_result = pd.read_csv("../results/csv/knn_cross_validate_result.csv")
knn_cross_validate_result.columns=["criteria", "score"]
knn_cross_validate_result


# KNN Classification Report:

# In[4]:


{
    "tags": [
        "hide-input"
    ]
}
knn_classification_report= pd.read_csv("../results/csv/knn_classification_report.csv")
knn_classification_report.columns.values[0]="index"
knn_classification_report


# ### Decision Tree
# A decision tree is a decision support tool that uses a tree-like model of decisions and their 
# possible consequences, including chance event outcomes, resource costs, and utility
# The goal of using a Decision Tree is to create a training model that can use to predict 
# the class or value of the target variable by learning simple decision rules inferred 
# from prior data(training data).

# ```{figure-md} f3
# <img src="../results/figures/dt_accuracy.png" alt="num" class="bg-primary mb-1" width="500px">
# 
# A plot reveals the relationship between deepth and corresponding accuracy
# ```

# As shown in the [fig.3](f3), the best depth of the Decision Tree is around small. We can confirm that the best value of the depth is 5 after tuning the hyperparameter and calculating the accuracy.

# ### Decision Tree final model & evaluation

# After training the model, we obtain the Cross Validation score, as well as the precision, recall, f1-score and support.

# DT Cross Validation Result:

# In[5]:


{
    "tags": [
        "hide-input"
    ]
}
dt_cross_validate_result = pd.read_csv("../results/csv/dt_cross_validate_result.csv")
dt_cross_validate_result.columns=["criteria", "score"]
dt_cross_validate_result


# DT Classification Report:

# In[6]:


{
    "tags": [
        "hide-input"
    ]
}
dt_classification_report = pd.read_csv("../results/csv/dt_classification_report.csv")
dt_classification_report.columns.values[0]="index"
dt_classification_report


# ### Support Vector Machine
# SVM or Support Vector Machine is a linear model for classification and regression problems. 
# It can solve linear and non-linear problems and work well for many practical problems. 
# The idea of SVM is simple: The algorithm creates a line or a hyperplane which separates the 
# data into classes.

# ### SVM training model Jaccard Score, final model and evaluation

# SVM Classification Report:

# In[7]:


{
    "tags": [
        "hide-input"
    ]
}
svm_classification_report= pd.read_csv("../results/csv/svm_classification_report.csv")
svm_classification_report.columns.values[0]="index"
svm_classification_report


# ### Logistic Regression
# Logistic Regression is a "Supervised machine learning" algorithm that can be used to model the probability of a certain class or event. It is used when the data is linearly separable and the outcome is binary or dichotomous in nature. That means Logistic regression is usually used for Binary classification problems.

# ### Logistic Regression training model Jaccard Score, final model and evaluation
# 

# LR Classification Report:

# In[8]:


{
    "tags": [
        "hide-input"
    ]
}
lr_classification_report= pd.read_csv("../results/csv/lr_classification_report.csv")
lr_classification_report.columns.values[0]="index"
lr_classification_report

