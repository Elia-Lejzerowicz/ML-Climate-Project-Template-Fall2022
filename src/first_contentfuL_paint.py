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
