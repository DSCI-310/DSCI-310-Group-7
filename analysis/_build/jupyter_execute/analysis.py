#!/usr/bin/env python
# coding: utf-8

# # Methods & Results
# We are going to use multiple analysis to classify the type of the animals using 16 variables including hair, feathers, eggs, milk, airborne, aquatic, predator, toothed, backbone, breathes, venomous, fins, legs, tail, domestic, catsize as our predictors. To predict the class of a new observation, the algorithms of each type will be further explained before implementation.

# In[1]:


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


# In[ ]:


# reading the data as a csv from the uci web server, with header = false as the data contains no header
# Adding column names to the data
colm = ["animalName", "hair", "feathers", "eggs", "milk", "airborne", "aquatic", 
        "predator", "toothed", "backbone", "breathes", "venomous", "fins", 
        "legs", "tail", "domestic", "catsize", "type"]

zoo_data = pre_process("https://archive.ics.uci.edu/ml/machine-learning-databases/zoo/zoo.data", colm)


# The first thing is to import the data. The data set is downloaded from [UCI repository]("https://archive-beta.ics.uci.edu/ml/datasets/zoo"). It is then saved as a csv file in this project repository. Some exploratory data analysis needs to be run before running the actual analyses on the data set. Here is a preview of pre-processed data set:

# In[9]:


zoo_data = pd.read_csv("../results/csv/head.csv")
zoo_data


# In[3]:


#uncomment this line if the server is down for fetching the data
#zoo_data = pd.read_csv("./data/zoo.csv")
zoo_data.head()


# In[4]:


# saving the data as a csv file in our data directory
# zoo_data.to_csv(r'./data/zoo.csv')


# In[5]:


# Check if there are missing values
print("Whether the dataset contains missing value: " + str(zoo_data.isna().any().any()))


# In[6]:


# drop the first column
#zoo_data = zoo_data.drop(zoo_data.columns[[0,1]], axis=1)


# Below is a summary of dataset:

# In[10]:


# Create a summary of the data set, including descriptive statistics
zoo_data.describe()


# As there aren't missing values in the data set, we can clearly deduce that the data set is clean according to the data summary we generated above. Since most features are binary and categorical, there is no need to do normalization and standardization.

# :::{figure-md} f1
# <img src="../results/figures/fig1.png" alt="num" class="bg-primary mb-1" width="800px">
# 
# A summary table of the data set
# :::

# As shown in [fig.1](f1), the histograms of each feature are generated. The ones with skewed distribution might be more decisive in the prediction. However, since the data set is relatively small, all the features except the `animalName` are going to be used to predict. In the next part, we are going to split the data, into the training set and testing set. After that, different classification models will be trained and evaluated.

# ## Classification
# Now we will use the training set to build an accurate model, whereas the testing set is used to report the accuracy of the models. Here is a list of algorithms we will use in the following section:
# 
# K Nearest Neighbor(KNN)
# <br>
# Decision Tree
# <br>
# Support Vector Machine
# <br>
# Logistic Regression

# In[9]:


# extracting the feature that will predict
feature = zoo_data[["hair", "feathers", "eggs", "milk", "airborne", 
                   "aquatic", "predator", "toothed", "backbone", "breathes", 
                   "venomous", "fins", "legs", "tail", "domestic", "catsize"]]
# making it as a X
X = feature
X.head()


# In[10]:


# taking the y values, the type
y = zoo_data['type']
y[0:5]


# The training set and test set are splitted.

# In[11]:


# splitting the dataset 80-20 for train and test
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print('Training set:', X_train.shape,  y_train.shape)
print('Test set:', X_test.shape,  y_test.shape)


# ### KNN
# KNN captures the idea of similarity (sometimes called distance, proximity, or closeness) 
# with some basic mathematics we might have learned earlier. Basically in terms of geometry we can always calculate the distance between points on a graph. Similarly, using KNN we can group similar points together and predict the target with our feature variables(x).

# In[12]:


#training the model for different set of K values and finding the best K value
Ks = 81
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))
ConfustionMx = [];
for n in range(1,Ks):
    
    #Train Model and Predict  
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat=neigh.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)

std_acc = stdAcc(yhat,y_test,Ks)
mean_acc


# :::{figure-md} f2
# <img src="../results/figures/k_accuracy.png" alt="num" class="bg-primary mb-1" width="500px">
# 
# A plot reveals the relationship between K and corresponding accuracy
# :::

# In[13]:


# plotting the accuracy for different K values
# plt.plot(range(1,Ks),mean_acc,'g')
# plt.fill_between(range(1,Ks), mean_acc - 1 * std_acc, mean_acc + 1 * std_acc, alpha=0.10)
# plt.legend(('Accuracy ', '+/- 3xstd'))
# plt.ylabel('Accuracy ')
# plt.xlabel('Number of Neighbors (K)')
# plt.title('fig.2 Number of Neighbors vs. Accuracy')
# plt.tight_layout()
line_plot(Ks, mean_acc, std_acc, "Number of Neighbors (K)", "Accuracy", "fig.2 Number of Neighbors vs. Accuracy")
plt.show()


# In[14]:


print("The best accuracy was with the values", mean_acc.max(), "with k=", mean_acc.argmax()+1 )


# In[15]:


# Finding the K value using Grid Search
knn = KNeighborsClassifier()
k_vals = list(range(1, 21))
param_grid = dict(n_neighbors=k_vals)

para_optimize(knn, param_grid, 3, X_train, y_train)


# ### KNN final model & Evaluation

# In[16]:


# as the best accuracy was with K = 1
# using K = 1 for the final KNN model
# Final KNN model is here used the splited test part to train again for better training, and better prediction
# KNN evaluation is also here scroll through the output
final_knn_model = finalModel("KNN", 1, X_train, X_test, y_train, y_test, X, y)


# KNN Cross Validation Result:

# In[34]:


knn_cross_validate_result = pd.read_csv("../results/csv/knn_cross_validate_result.csv")
knn_cross_validate_result


# KNN Classification Report:

# In[40]:


knn_classification_report= pd.read_csv("../results/csv/knn_classification_report.csv")
knn_classification_report


# ### Decision Tree
# A decision tree is a decision support tool that uses a tree-like model of decisions and their 
# possible consequences, including chance event outcomes, resource costs, and utility
# The goal of using a Decision Tree is to create a training model that can use to predict 
# the class or value of the target variable by learning simple decision rules inferred 
# from prior data(training data).

# In[17]:


Ks = 50
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))
ConfustionMx = [];
for n in range(1,Ks):
    # Train Model and Predict  
    decTree = DecisionTreeClassifier(criterion="entropy", max_depth = n)
    decTree.fit(X_train,y_train)
    yhat=decTree.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)

std_acc = stdAcc(yhat,y_test,Ks)
mean_acc


# :::{figure-md} f3
# <img src="../results/figures/dt_accuracy.png" alt="num" class="bg-primary mb-1" width="500px">
# 
# A plot reveals the relationship between deepth and corresponding accuracy
# :::

# In[18]:


# plt.plot(range(1,Ks),mean_acc,'g')
# plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
# plt.legend(('Accuracy ', '+/- 3xstd'))
# plt.ylabel('Accuracy ')
# plt.xlabel('Max_Depth')
# plt.title('fig.3 Max_depth vs. Accuracy')
# plt.tight_layout()
line_plot(Ks, mean_acc, std_acc, "Maximum Depth", "Accuracy", "fig.2 Relationship between Number of Neighbors and Accuracy")
plt.show()


# In[19]:


print("The best accuracy was with the values", mean_acc.max(), "with max_depth =", mean_acc.argmax()+1)


# ### Decision Tree final model & evaluation

# In[20]:


# As Best is max depth = 5
# using max depth = 5 for the final decision tree
# Final decision tree is here used the split test part to train again for better training, and better prediction
# DT evaluation is also here scroll through the output
Final_dec_Tree = finalModel("DT", 5, X_train, X_test, y_train, y_test, X, y)


# DT Cross Validation Result:

# In[36]:


dt_cross_validate_result = pd.read_csv("../results/csv/dt_cross_validate_result.csv")
dt_cross_validate_result


# DT Classification Report:

# In[37]:


dt_classification_report = pd.read_csv("../results/csv/dt_classification_report.csv")
dt_classification_report


# ### Support Vector Machine
# SVM or Support Vector Machine is a linear model for classification and regression problems. 
# It can solve linear and non-linear problems and work well for many practical problems. 
# The idea of SVM is simple: The algorithm creates a line or a hyperplane which separates the 
# data into classes.

# ### SVM training model Jaccard Score, final model and evaluation

# In[21]:


#Final SVM is here used the splited test part to train again for better training, and better prediction
#svm evaluation train and final model is also here scroll through the output
svec = finalModel("SVM", -1, X_train, X_test, y_train, y_test, X, y)


# SVM Classification Report:

# In[41]:


svm_classification_report= pd.read_csv("../results/csv/svm_classification_report.csv")
svm_classification_report


# ### Logistic Regression
# Logistic Regression is a "Supervised machine learning" algorithm that can be used to model the probability of a certain class or event. It is used when the data is linearly separable and the outcome is binary or dichotomous in nature. That means Logistic regression is usually used for Binary classification problems.

# ### Logistic Regression training model Jaccard Score, final model and evaluation
# 

# In[22]:


# final LR model is here used the splited test part to train again for better training, and better prediction
# LR evaluation train and final model is also here scroll through the output
LR = finalModel("LR", -1, X_train, X_test, y_train, y_test, X, y)


# LR Classification Report:

# In[43]:


lr_classification_report= pd.read_csv("../results/csv/lr_classification_report.csv")
lr_classification_report

