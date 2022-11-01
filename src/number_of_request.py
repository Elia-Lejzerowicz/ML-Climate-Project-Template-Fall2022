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
print("")
print("list of the selected features for the algorithm")
print(importance_nbRequest_list)

