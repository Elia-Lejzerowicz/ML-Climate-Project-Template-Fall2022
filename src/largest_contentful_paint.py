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


