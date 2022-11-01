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


