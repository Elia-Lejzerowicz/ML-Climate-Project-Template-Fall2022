
# Create a new dataset to hold the new recommendation

new_dataset = urlscore3

for recommendations in First_Recommendation:


   for i in range(len(new_dataset)):

     change = First_Recommendation[recommendations] 


     #Increase the value to a 100
     
     new_dataset.loc[i,change] = 100



# Retrain thee model with the recommenations

y = new_dataset["classe_domSize"]
X = new_dataset.drop(columns= classes_to_drop + ["classe_domSize"] + ["domSize"])

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.25, random_state = 42)

forest = RandomForestClassifier()

random_rf = RandomizedSearchCV(estimator = forest, param_distributions = grid, n_iter = 60)
                               
random_rf.fit(X_train, y_train)


print(f'model score on training data : {random_rf.score(X_train, y_train)}')
print(f'model score on testing data: {random_rf.score(X_test, y_test)}')



#Look at the change from the recommendations

old_dataset_DomSize = urlscore3[['DomSize']]

new_dataset_DomSize  = new_dataset[['DomSize']]

comparaison = pd.concat([old_dataset_DomSize, new_dataset_DomSize ], axis='col') 

mean_old = comparaison.iloc[:, 0]
mean_new = comparaison.iloc[:, 1]

change = mean_new  - mean_old 

print(change)



# Diaplay thee change 


import matplotlib.pyplot as plt
import numpy as np
  
# create data
x1 = [i for i in range(len( new_dataset))]
y1 = comparaison.iloc[:, 0]

x2 = [i for i in range(len( new_dataset))]
y2 = comparaison.iloc[:, 1]

  
# plot lines
plt.plot(x1, y1, label = "Old Domsize")
plt.plot(x2, y2, label = "New Domsize")

plt.legend()
plt.show()