#!/usr/bin/env python
# coding: utf-8

# # Prediction on Animal Type

# ## Summary
# The data set we will be using is Zoo (1990) provided by UC Irvine Machine Learning Repository. It stores data with 7 classes of animals and their related characteristics including animal name, hair, feathers and other attributes. In this project, we will use classification as our method to predict a most likely type of a given animal.
# ## Introduction
# The earth is an amazing planet that cultivates branches of animals. In general, scholars split them into 12 classes including mammals, birds, reptiles, amphibians, fishes, insects, crustaceans, arachnids, echinoderms, worms, mollusks and sponges(BioExploer.net., 2022). The traditional way in animal classification is manually identifying the characteristics and attributing it the mostly close class (Manohar, Sharath, & Kumar, 2016). However, it is tedious and time consuming, especially when the data set is very huge. A question hereby comes to us, if we can apply K-nearest neighbors (KNN) algorithms in predicting the type an animal belongs to given its related characteristics, such as hair, feathers, etc.? Therefore, in this project, we will show how we use KNN to do classification in animals based on data set Zoo(1990) which contains 1 categorical attribute, 17 Boolean-valued attributes and 1 numerical attribute. The categorical attribute appears to be the class attribute. Detailed breakdowns are as follows:
# 1. animal name: Unique for each instance 
# 2. hair: Boolean 
# 3. feathers: Boolean 
# 4. eggs: Boolean 
# 5. milk: Boolean 
# 6. airborne: Boolean 
# 7. aquatic: Boolean 
# 8. predator: Boolean 
# 9. toothed: Boolean 
# 10. backbone: Boolean 
# 11. breathes: Boolean 
# 12. venomous: Boolean 
# 13. fins: Boolean 
# 14. legs: Numeric (set of values: {0,2,4,5,6,8}) 
# 15. tail: Boolean 
# 16. domestic: Boolean 
# 17. catsize: Boolean 
# 18. type: Numeric (integer values in range [1,7])
# 
# ## Methods & Results
# We are going to use multiple analysis to classify the type of the animals using 16 variables including hair, feathers, eggs, milk, airborne, aquatic, predator, toothed, backbone, breathes, venomous, fins, legs, tail, domestic, catsize as our predictors. To predict the class of a new observation, the algorithms of each type will be further explained before implementation.

# In[1]:


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


# In[2]:


# reading the data as a csv from the uci web server, with header = false as the data contains no header
# Adding column names to the data
colm = ["animalName", "hair", "feathers", "eggs", "milk", "airborne", "aquatic", 
        "predator", "toothed", "backbone", "breathes", "venomous", "fins", 
        "legs", "tail", "domestic", "catsize", "type"]

zoo_data = pre_process("https://archive.ics.uci.edu/ml/machine-learning-databases/zoo/zoo.data", colm)


# In[3]:


#uncomment this line if the server is down for fetching the data
#zoo_data = pd.read_csv("./data/zoo.csv")
zoo_data.head()


# The first thing is to import the data. The data set is downloaded from [UCI repository]("https://archive-beta.ics.uci.edu/ml/datasets/zoo"). It is then saved as a csv file in this project repository. Some exploratory data analysis needs to be run before running the actual analyses on the data set.

# In[4]:


# saving the data as a csv file in our data directory
# zoo_data.to_csv(r'./data/zoo.csv')


# In[5]:


# Check if there are missing values
print("Whether the dataset contains missing value: " + str(zoo_data.isna().any().any()))


# In[6]:


# drop the first column
#zoo_data = zoo_data.drop(zoo_data.columns[[0,1]], axis=1)


# In[7]:


# Create a summary of the data set, including descriptive statistics
zoo_data.describe()


# After checking whether there are missing values in the data set, we can clearly deduce that the data set is clean according to the data summary we generated above. Since most features are binary and categorical, there is no need to do normalization and standardization.

# In[8]:


# Create a visualization of the dataset
zoo_data.hist(figsize=(15,15));
plt.suptitle("fig.1 Histogram of all features");


# As shown in fig.1, the histograms of each feature are generated. The ones with skewed distribution might be more decisive in the prediction. However, since the data set is relatively small, all the features except the `animalName` are going to be used to predict. In the next part, we are going to split the data, into the training set and testing set. After that, different classification models will be trained and evaluated.

# ### Classification
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


# In[11]:


# splitting the dataset 80-20 for train and test
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print('Training set:', X_train.shape,  y_train.shape)
print('Test set:', X_test.shape,  y_test.shape)


# #### KNN
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


# #### KNN final model & Evaluation

# In[16]:


# as the best accuracy was with K = 1
# using K = 1 for the final KNN model
# Final KNN model is here used the splited test part to train again for better training, and better prediction
# KNN evaluation is also here scroll through the output
final_knn_model = finalModel("KNN", 1, X_train, X_test, y_train, y_test, X, y)


# #### Decision Tree
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


# #### Decision Tree final model & evaluation

# In[20]:


# As Best is max depth = 5
# using max depth = 5 for the final decision tree
# Final decision tree is here used the split test part to train again for better training, and better prediction
# DT evaluation is also here scroll through the output
Final_dec_Tree = finalModel("DT", 5, X_train, X_test, y_train, y_test, X, y)


# #### Support Vector Machine
# SVM or Support Vector Machine is a linear model for classification and regression problems. 
# It can solve linear and non-linear problems and work well for many practical problems. 
# The idea of SVM is simple: The algorithm creates a line or a hyperplane which separates the 
# data into classes.

# #### SVM training model Jaccard Score, final model and evaluation

# In[21]:


#Final SVM is here used the splited test part to train again for better training, and better prediction
#svm evaluation train and final model is also here scroll through the output
svec = finalModel("SVM", -1, X_train, X_test, y_train, y_test, X, y)


# #### Logistic Regression
# Logistic Regression is a "Supervised machine learning" algorithm that can be used to model the probability of a certain class or event. It is used when the data is linearly separable and the outcome is binary or dichotomous in nature. That means Logistic regression is usually used for Binary classification problems.

# #### Logistic Regression training model Jaccard Score, final model and evaluation
# 

# In[22]:


# final LR model is here used the splited test part to train again for better training, and better prediction
# LR evaluation train and final model is also here scroll through the output
LR = finalModel("LR", -1, X_train, X_test, y_train, y_test, X, y)


# </br>
# 
# ## Discussion

# After analyzing all the different 4 models K Nearest Neighbor(KNN), Decision Tree, Support Vector Machine and Logistic Regression, we found KNN is best to predict the animal type here. As you have seen in the model evaluation tables before, for accuracy KNN is the best, the second-best is decision tree method and following by Support Vector Machine and Logistic Regression. The result of KNN was expected as KNN is the best in grouping similar data points together and giving the best prediction results. Predicting the correct animal type with the highest accuracy have a huge impact on identifying animal types. These models can be used to identify animal types instantly for example if someone saw/discovered an animal and the type is not identified then they can feed all the characteristics fields to the model. The model can predict the animal type accurately, which is way more accurate than identifying and classifying the animal based on common sense. Thus our model can increase the research potential in many fields but not just limited to Marine Science, Animal Science, Forestry, and etc. This might lead to a future question in which how we are going to maintain the accuracy of predictions when working with more diverse groups of animals. Another possible aspect of this can be how some attributes of animals will relate to each other, for instance, relation between animals which has teeth vs predator. Furthermore, how we are going to use those relations to predict behaviors and attributes of newly discovered animals and how we are going to make our perceptions on animals even more detailed. These models and their advancements will not only widen our knowledge in terms of animal biology but will also let us find all other possible relations within the nature in a much more efficient way.

# ## Citation
# BioExplorer.net. (2022, February 18). Types of Animals. Bio Explorer. https://www.bioexplorer.net/animals/.
# 
# N. Manohar, Y. H. Sharath Kumar and G. H. Kumar, \"Supervised and unsupervised learning in animal classification," *2016 International Conference on Advances in Computing, Communications and Informatics (ICACCI)*, 2016, pp. 156-161, doi: 10.1109/ICACCI.2016.7732040.
# 
# Tiffany Timbers, T. C. (2022, February 13). Data science: A first Introduction. Retrieved February 18, 2022, from https://datasciencebook.ca/. 
# 
# Zoo. (1990). UCI Machine Learning Repository.
# 
# Towards Data Science - Medium Articles.
# 
# IBM Data Science Resources.

# In[ ]:




