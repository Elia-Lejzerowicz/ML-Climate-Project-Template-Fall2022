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
Footer
© 2022 GitHub, Inc.
Footer navigation
Terms
