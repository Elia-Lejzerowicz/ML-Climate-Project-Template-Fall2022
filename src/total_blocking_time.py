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
