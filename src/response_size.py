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
