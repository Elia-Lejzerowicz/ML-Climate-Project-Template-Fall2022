Journal - Date entries:

1) Week of 12/09:

- Project research: 
    Looked for ideas of project

2) Week of 19/09:

- Project definition
- Project proposal
- Data gathering:  
    - Gathered data based on the analysis from google lighthouse and greenIT extension
    - 500 urls of websites analyzed


3) Week of 26/09:

Data gathering, construction + exploration:

- merged the data from the different ressources together (google lighthouse + greenIT)
- created 5 differents dataset, each one with different classes:
    - 2/3/4 or 5 classes to do the classification problems

4) Week of 3/10:

Worked on DomSize classification problem:
- Random Forest Classification prediction 
- Feature selection
- Feature importance 
- Tried the same code with 2, 3, 4 and 5 classes
    - used 3 classes for the classification as it is what yields highest resuts


5) Week of 10/10:

- Hyperparamter tuning with random search for DomSize
- Random Forest classification problems for: 
    - Number of request
    - Size of the page
    - First Contentful Paint


6) Week of 17/10:

- Random Forest classification + Hyperparamter tuning + feature selection + feature importance for: 
  
    - Speed Index
    - Largest Contentful Paint
    - Time to Interactive
    - Total Blocking Time
    - Cumulative Layout Shift

7) Week of 24/10:

- Started to code the recommendation Algorithm:

Algorithm that recommend the best practices from the 49 practices to implement in order to obtain a better Eco-index score and Google Lighthouse Performance score.

    The algorithm is based on a combination of:
    
        - feature importance
        - complexity 
        - and metric’s weight


8) Week of 31/10:

- Finished coding the recommendation Algorithm


9) Week of 7/11:

 - Coded the "apply recommenaton algoritm":
    - Firt try with only DomSize
   It creates a new dataset by changing the ol dataset based on the recommendations 
   
   It then retrains the model with the changed values and predicts the new DomSize
   It calculates the, hopeful, increase in the score of the Domsize
   
   + Display the graph



