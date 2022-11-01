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

