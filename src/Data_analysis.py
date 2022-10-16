
#data treatment
import pandas as pd
from pandas import DataFrame
import numpy as np
import random
from pandas_profiling import ProfileReport

#vizualisation
import matplotlib.pyplot as plt 
import seaborn as sns
from itertools import chain
import dash_bootstrap_components as dbc
from matplotlib.legend import Legend


#Classification ML
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn import tree
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import RandomizedSearchCV

#SHAP
import shap

#logistic regression:
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#other importance calc
from sklearn.inspection import permutation_importance



urlscore = pd.read_csv("Data_website.csv")
print(urlscore.head(10))



#retrieve the columns
column = urlscore.columns

#those are thre classes to drop each time
classes_to_drop = ['classe_nbRequest', 'classe_responsesSize', 'classe_domSize',
       'classe_largestContentfulPaint', 'classe_cumulativeLayoutShift',
       'classe_firstContentfulPaint', 'classe_speedIndex',
       'classe_totalBlockingTime', 'classe_interactive']

#Grid for hyperparameter tuning
#hyperparamter tuning
grid = {'n_estimators': [5,10,20,30,50,100,200] ,
               
            'max_features': ['auto', 'sqrt','log2'],

            'max_depth': [3, 5, 7, 9,12] ,

            'min_samples_split': [2,4, 6,8, 10,15,20,30] ,

            'min_samples_leaf': [1, 3, 4,6,10,15,17],

    
       }

    
# n_estimators: This is the number of trees to use  
# max_features =  The number of features to consider when looking for the best split
# max_depth: The max depth at each decsision tree
# min_samples_split= minimum number to split at each node
# min_samples_leaf =  minimum sample number that can be stored in a leaf node
# bootstrap =method used to sample data points
#'criterion': ['gini', 'entropy'] but better without

print("The dataset contains", {urlscore.shape[0]}," samples and", {urlscore.shape[1]}," features")







#### DOMSIZE ####

y = urlscore["classe_domSize"]
X = urlscore.drop(columns= classes_to_drop + ["classe_domSize"] + ["domSize"])

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.25, random_state = 42)

forest = RandomForestClassifier()

random_rf = RandomizedSearchCV(estimator = forest, param_distributions = grid, n_iter = 60)
                               
random_rf.fit(X_train, y_train)

print ('Best Parameters: ', random_rf.best_params_, ' \n')

print("reesults before feature selectio:")
print(f'model score on training data : {random_rf.score(X_train, y_train)}')
print(f'model score on testing data: {random_rf.score(X_test, y_test)}')

importances_dom = random_rf.best_estimator_.feature_importances_


#####################
# Feature selection #
#####################

print("results after feature selection:")

#get a list of the features importances
importances_dom_list = feature_selection(importances_dom, X_train)


#The dependant variables are now selected 
X = X[importances_dom_list]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.25, random_state = 42)

forest = RandomForestClassifier()

random_rf = RandomizedSearchCV(estimator = forest, param_distributions = grid, n_iter = 60)
                               
random_rf.fit(X_train, y_train)

print("results before feature selection:")
print ('Best Parameters: ', random_rf.best_params_, ' \n')


print(f'model score on training data : {random_rf.score(X_train, y_train)}')
print(f'model score on testing data: {random_rf.score(X_test, y_test)}')


importances_dom = random_rf.best_estimator_.feature_importances_


#Plot the features importances
plt.figure(figsize=(20,20))
plt.title('Feature Importances: domSize')

(pd.Series(importances_dom, index=X_train.columns)
   .nlargest(30)
   .plot(kind='barh')) 


#get a list of the features that will be selected for the algorithm
importance_domSize_list = get_feature_importance(importances_dom, X_train)
print(importance_domSize_list)
