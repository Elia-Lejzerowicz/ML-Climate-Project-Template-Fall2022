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
