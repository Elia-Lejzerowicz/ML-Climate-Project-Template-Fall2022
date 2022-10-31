
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



urlscore3 = pd.read_csv("urlscore3.csv")
print(urlscore3.head(10))



#retrieve the columns
column = urlscore3.columns

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

print("The dataset contains", {urlscore3.shape[0]}," samples and", {urlscore3.shape[1]}," features")


#function to get the feature importance to do the feature selection.
#This functions create a list with with all the features importances that the model considers higher 
#than the random variable
#this allows to retrain the model without the features that are not important

def feature_selection(importance_metric, X_train_number):
    
    r = pd.Series(importance_metric, index=X_train_number.columns)
    r = r.sort_values(axis=0, ascending=False)
    r = r.index.tolist()
    features = []
    for e in r:
        if e == 'RandomNumber':
            break
        features.append(e)
    
    return features

#Fucntion to get the most important features
#Here, we consider that it is an important feature when its greater than 1, but that can be changed ! 

def get_feature_importance(importance_metric, X_train_number):

    feat = pd.Series(importance_metric*100, index=X_train_number.columns)
    feat_s = feat.sort_values(axis=0, ascending=False)
    feat = feat_s.index.tolist()
    
    features = []
    for values in feat:
        if feat_s[values] > 1 :
            features.append(values)
            
    
    return features




#### DOMSIZE ####

y = urlscore3["classe_domSize"]
X = urlscore3.drop(columns= classes_to_drop + ["classe_domSize"] + ["domSize"])

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


#### Number of Requests #########

y2 = urlscore3["classe_nbRequest"]
X2 = urlscore3.drop(columns= classes_to_drop+ ['httpRequests'], axis = 1 )


X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, stratify=y2,test_size=0.3, random_state=42)


forest2 = RandomForestClassifier()

random_rf2 = RandomizedSearchCV(estimator = forest2, param_distributions = grid, n_iter = 60)
                               
random_rf2.fit(X_train2, y_train2)

print("Results before feature selection:")
print ('Best Parameters: ', random_rf2.best_params_, ' \n')
print(f'model score on training data: {random_rf2.score(X_train2, y_train2)}')
print(f'model score on testing data: {random_rf2.score(X_test2, y_test2)}')

importances_nbRequest = random_rf2.best_estimator_.feature_importances_

#Plot the features importances
plt.figure(figsize=(20,20))
plt.title('Feature Importances: NumberOfRequest')

(pd.Series(importances_nbRequest, index=X2.columns)
   .nlargest(20)
   .plot(kind='barh'))


#####################
# Feature selection #
#####################
print("")
print("Results after feature selection:")

#get a list of the features importances
importances_nbRequest_list = feature_selection(importances_nbRequest, X_train2)

#The dependant variables are now selected 
X2 = X2[importances_nbRequest_list]

#re-appplication of the model
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, stratify=y2,test_size=0.3, random_state=42)

forest2 = RandomForestClassifier()

random_rf2 = RandomizedSearchCV(estimator = forest2, param_distributions = grid, n_iter = 60)
                               
random_rf2.fit(X_train2, y_train2)

print ('Best Parameters: ', random_rf2.best_params_, ' \n')
print(f'model score on training data: {random_rf2.score(X_train2, y_train2)}')
print(f'model score on testing data: {random_rf2.score(X_test2, y_test2)}')

importances_nbRequest = random_rf2.best_estimator_.feature_importances_
 

#get a list of the features that will be selected for the algorithm
importance_nbRequest_list = get_feature_importance(importances_nbRequest, X_train2)
print("")
print("list of the selected features for the algorithm")
print(importance_nbRequest_list)


###### Size of the page (responsesSize) ##########

y3 = urlscore3["classe_responsesSize"]
X3 = urlscore3.drop(columns= classes_to_drop)

X_train3, X_test3, y_train3, y_test3 = train_test_split(X3, y3, stratify=y3,test_size=0.2, random_state=42)


forest3 = RandomForestClassifier()

random_rf3 = RandomizedSearchCV(estimator = forest3, param_distributions = grid, n_iter = 60)
                               
random_rf3.fit(X_train3, y_train3)

print ('Best Parameters: ', random_rf3.best_params_, ' \n')

print(f'model score on training data: {random_rf3.score(X_train3, y_train3)}')
print(f'model score on testing data: {random_rf3.score(X_test3, y_test3)}')

importances_responsesSize = random_rf3.best_estimator_.feature_importances_

#Plot the features importances
plt.figure(figsize=(20,20))
plt.title('Feature Importances: responsesSize')

(pd.Series(importances_responsesSize, index=X3.columns)
   .nlargest(20)
   .plot(kind='barh')) 

#####################
# Feature selection #
#####################

print("")
print("Results after feature selection:")

#get a list of the features importances
importances_responsesSize_list = feature_selection(importances_responsesSize, X_train3)

#The dependant variables are now selected 
X3 = X3[importances_responsesSize_list]

X_train3, X_test3, y_train3, y_test3 = train_test_split(X3, y3, stratify=y3,test_size=0.2, random_state=42)


forest3 = RandomForestClassifier()

random_rf3 = RandomizedSearchCV(estimator = forest3, param_distributions = grid, n_iter = 60)
                               
random_rf3.fit(X_train3, y_train3)

print ('Best Parameters: ', random_rf3.best_params_, ' \n')

print(f'model score on training data: {random_rf3.score(X_train3, y_train3)}')
print(f'model score on testing data: {random_rf3.score(X_test3, y_test3)}')

importances_responsesSize = random_rf3.best_estimator_.feature_importances_

#get a list of the features that will be selected for the algorithm
importances_responsesSize_list = get_feature_importance(importances_responsesSize, X_train3)
print("")
print("list of the selected features for the algorithm")
print(importances_responsesSize_list)


######### Largest Contenful Paint ###############

y4 = urlscore3["classe_largestContentfulPaint"]
X4 = urlscore3.drop(columns= classes_to_drop )


X_train4, X_test4, y_train4, y_test4 = train_test_split(X4, y4, stratify=y4,test_size=0.2, random_state=42)


forest4 = RandomForestClassifier()

random_rf4 = RandomizedSearchCV(estimator = forest4, param_distributions = grid, n_iter = 60)
                               
random_rf4.fit(X_train4, y_train4)

print("Results before feature selection:")
print ('Best Parameters: ', random_rf4.best_params_, ' \n')


print(f'model score on training data: {random_rf4.score(X_train4, y_train4)}')
print(f'model score on testing data: {random_rf4.score(X_test4, y_test4)}')

importances_largestContentfulPaint = random_rf4.best_estimator_.feature_importances_

#Plot the features importances
plt.figure(figsize=(20,20))
plt.title('Feature Importances: largestContentfulPaint')

(pd.Series(importances_largestContentfulPaint, index=X_train4.columns)
   .nlargest(20)
   .plot(kind='barh')) 


#####################
# Feature selection #
#####################
print("")
print("Results after feature selection:")

#get a list of the features importances
importances_largestContentfulPaint_list = feature_selection(importances_largestContentfulPaint, X_train4)

#The dependant variables are now selected 
X4 = X4[importances_largestContentfulPaint_list]

X_train4, X_test4, y_train4, y_test4 = train_test_split(X4, y4, stratify=y4,test_size=0.2, random_state=42)


forest4 = RandomForestClassifier()

random_rf4 = RandomizedSearchCV(estimator = forest4, param_distributions = grid, n_iter = 60)
                               
random_rf4.fit(X_train4, y_train4)

print(f'model score on training data: {random_rf4.score(X_train4, y_train4)}')
print(f'model score on testing data: {random_rf4.score(X_test4, y_test4)}')

importances_largestContentfulPaint = random_rf4.best_estimator_.feature_importances_


#get a list of the features that will be selected for the algorithm
importance_largestContentfulPaint_list = get_feature_importance(importances_largestContentfulPaint, X_train4)
print("")
print("list of the selected features for the algorithm")
print(importance_largestContentfulPaint_list)



######### Cumulative layout Shift ##################

y5 = urlscore3["classe_cumulativeLayoutShift"]
X5 = urlscore3.drop(columns= classes_to_drop )

X_train5, X_test5, y_train5, y_test5 = train_test_split(X5, y5, stratify=y5,test_size=0.2, random_state=42)

forest5 = RandomForestClassifier()

random_rf5 = RandomizedSearchCV(estimator = forest5, param_distributions = grid, n_iter = 60)
                               
random_rf5.fit(X_train5, y_train5)

print("Results before feature selection:")
print ('Best Parameters: ', random_rf5.best_params_, ' \n')


print(f'model score on training data: {random_rf5.score(X_train5, y_train5)}')
print(f'model score on testing data: {random_rf5.score(X_test5, y_test5)}')

importances5 = random_rf5.best_estimator_.feature_importances_

#Plot the features importances
plt.figure(figsize=(20,20))
plt.title('Feature Importances: cumulativeLayoutShift')

(pd.Series(importances5, index=X_train5.columns)
   .nlargest(20)
   .plot(kind='barh')) 


#####################
# Feature selection #
#####################
print("")
print("Results after feature selection:")

#get a list of the features importances
importances_cumulativeLayoutShift_list = feature_selection(importances5, X_train5)

#The dependant variables are now selected 
X5 = X5[importances_cumulativeLayoutShift_list]

X_train5, X_test5, y_train5, y_test5 = train_test_split(X5, y5, stratify=y5,test_size=0.2, random_state=42)

forest5 = RandomForestClassifier()

random_rf5 = RandomizedSearchCV(estimator = forest5, param_distributions = grid, n_iter = 60)
                               
random_rf5.fit(X_train5, y_train5)

print ('Best Parameters: ', random_rf5.best_params_, ' \n')


print(f'model score on training data: {random_rf5.score(X_train5, y_train5)}')
print(f'model score on testing data: {random_rf5.score(X_test5, y_test5)}')

importances_cumulativeLayoutShift = random_rf5.best_estimator_.feature_importances_


#get a list of the features that will be selected for the algorithm
importance_cumulativeLayoutShift_list = get_feature_importance(importances_cumulativeLayoutShift, X_train5)
print("")
print("list of the selected features for the algorithm")
print(importance_cumulativeLayoutShift_list)



##### First Contenful Paint ############

#without Feature Selection
y6 = urlscore3["classe_firstContentfulPaint"]
X6 = urlscore3.drop(columns= classes_to_drop )


X_train6, X_test6, y_train6, y_test6 = train_test_split(X6, y6, stratify=y6,test_size=0.2, random_state=42)


forest6 = RandomForestClassifier()

random_rf6 = RandomizedSearchCV(estimator = forest6, param_distributions = grid, n_iter = 60)
                               
random_rf6.fit(X_train6, y_train6)

print ('Best Parameters: ', random_rf6.best_params_, ' \n')


print(f'model score on training data: {random_rf6.score(X_train6, y_train6)}')
print(f'model score on testing data: {random_rf6.score(X_test6, y_test6)}')

importances6 = random_rf6.best_estimator_.feature_importances_

#Plot the features importances
plt.figure(figsize=(20,20))
plt.title('Feature Importances: firstContentfulPaint')

(pd.Series(importances6, index=X_train6.columns)
   .nlargest(20)
   .plot(kind='barh')) 


#####################
# Feature selection #
#####################

print("")
print("Results after feature selection:")

#get a list of the features importances
importances_firstContentfulPaint_list = feature_selection(importances6, X_train6)

#The dependant variables are now selected 
X6 = X6[importances_firstContentfulPaint_list]


X_train6, X_test6, y_train6, y_test6 = train_test_split(X6, y6, stratify=y6,test_size=0.2, random_state=42)


forest6 = RandomForestClassifier()

random_rf6 = RandomizedSearchCV(estimator = forest6, param_distributions = grid, n_iter = 60)
                               
random_rf6.fit(X_train6, y_train6)

print ('Best Parameters: ', random_rf6.best_params_, ' \n')


print(f'model score on training data: {random_rf6.score(X_train6, y_train6)}')
print(f'model score on testing data: {random_rf6.score(X_test6, y_test6)}')

importances6 = random_rf6.best_estimator_.feature_importances_

#get a list of the features that will be selected for the algorithm
importances_firstContentfulPaint_list = get_feature_importance(importances6, X_train6)
print("")
print("list of the selected features for the algorithm")
print(importances_firstContentfulPaint_list)


######### Speed Index #############

y7 = urlscore3["classe_speedIndex"]
X7 = urlscore3.drop(columns= classes_to_drop )


X_train7, X_test7, y_train7, y_test7 = train_test_split(X7, y7, stratify=y7,test_size=0.3, random_state=42)


forest7 = RandomForestClassifier()

random_rf7 = RandomizedSearchCV(estimator = forest7, param_distributions = grid, n_iter = 60)
                               
random_rf7.fit(X_train7, y_train7)

print("Results before feature selection:")
print ('Best Parameters: ', random_rf7.best_params_, ' \n')


print(f'model score on training data: {random_rf7.score(X_train7, y_train7)}')
print(f'model score on testing data: {random_rf7.score(X_test7, y_test7)}')

importances7 = random_rf7.best_estimator_.feature_importances_

#Plot the features importances
plt.figure(figsize=(20,20))
plt.title('Feature Importances: SpeedIndex')

(pd.Series(importances7, index=X_train7.columns)
   .nlargest(20)
   .plot(kind='barh')) 

#####################
# Feature selection #
#####################

print("")
print("Results after feature selection:")

#get a list of the features importances
importances_speedIndext_list = feature_selection(importances7, X_train7)

#The dependant variables are now selected 
X7 = X7[importances_speedIndext_list]

X_train7, X_test7, y_train7, y_test7 = train_test_split(X7, y7, stratify=y7,test_size=0.3, random_state=42)


forest7 = RandomForestClassifier()

random_rf7 = RandomizedSearchCV(estimator = forest7, param_distributions = grid, n_iter = 60)
                               
random_rf7.fit(X_train7, y_train7)

print ('Best Parameters: ', random_rf7.best_params_, ' \n')


print(f'model score on training data: {random_rf7.score(X_train7, y_train7)}')
print(f'model score on testing data: {random_rf7.score(X_test7, y_test7)}')

importances7 = random_rf7.best_estimator_.feature_importances_

#get a list of the features that will be selected for the algorithm
importances_speedIndext_list = get_feature_importance(importances7, X_train7)
print("")
print("list of the selected features for the algorithm")
print(importances_speedIndext_list)

############## Total Blocking Time ##################

y8 = urlscore3["classe_totalBlockingTime"]
X8 = urlscore3.drop(columns= classes_to_drop )


X_train8, X_test8, y_train8, y_test8 = train_test_split(X8, y8, stratify=y8,test_size=0.2, random_state=42)


forest8 = RandomForestClassifier()

random_rf8 = RandomizedSearchCV(estimator = forest8, param_distributions = grid, n_iter = 60)
                               
random_rf8.fit(X_train8, y_train8)

print("Results before feature selection:")
print ('Best Parameters: ', random_rf8.best_params_, ' \n')


print(f'model score on training data: {random_rf8.score(X_train8, y_train8)}')
print(f'model score on testing data: {random_rf8.score(X_test8, y_test8)}')

importances8 = random_rf8.best_estimator_.feature_importances_

#Plot the features importances
plt.figure(figsize=(20,20))
plt.title('Feature Importances: Total Blocking Time')

(pd.Series(importances8, index=X_train8.columns)
   .nlargest(20)
   .plot(kind='barh')) 



#get a list of the features that will be selected for the algorithm
importances_totalBlockingTime_list = get_feature_importance(importances8, X_train8)
print("")
print("list of the selected features for the algorithm")
print(importances_totalBlockingTime_list)

################ Time To Interactive #######################

y9 = urlscore3["classe_interactive"]
X9 = urlscore3.drop(columns= classes_to_drop )


X_train9, X_test9, y_train9, y_test9 = train_test_split(X9, y9, stratify=y9,test_size=0.2, random_state=42)


forest9 = RandomForestClassifier()

random_rf9 = RandomizedSearchCV(estimator = forest9, param_distributions = grid, n_iter = 60)
                               
random_rf9.fit(X_train9, y_train9)

print("Results before feature selection:")
print ('Best Parameters: ', random_rf9.best_params_, ' \n')


print(f'model score on training data: {random_rf9.score(X_train9, y_train9)}')
print(f'model score on testing data: {random_rf9.score(X_test9, y_test9)}')

importances9 = random_rf9.best_estimator_.feature_importances_

#Plot the features importances
plt.figure(figsize=(20,20))
plt.title('Feature Importances: interactive')

(pd.Series(importances9, index=X_train9.columns)
   .nlargest(20)
   .plot(kind='barh'))



#get a list of the features that will be selected for the algorithm
importances_interactive_list = get_feature_importance(importances9, X_train9)
print("")
print("list of the selected features for the algorithm")
print(importances_interactive_list)
