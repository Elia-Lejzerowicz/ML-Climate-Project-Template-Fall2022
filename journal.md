Journal - Date entries:

1) Week of 12/09:

- Project research: 
    Looked for ideas of project

2) Week of 19/09:

- Spoke with Accenture Technology in france to help them on a project on Machine Learning for climate
- Project definition + proposal
- Data gathering:  
    - Gathered data based on the analysis from google lighthouse and greenIT extension 
    
    GreenIT : https://github.com/cnumr/GreenIT-Analysis/blob/master/README.md
    GoogleLigthouse: https://developer.chrome.com/docs/lighthouse/overview/
    
      - 500 urls of websites analyzed:
    https://github.com/Accenture/EcoSonar
  

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
    
   It creates a new dataset by changing the old dataset based on the recommendations 
   
   It then retrains the model with the changed values and predicts the new DomSize
   
   It calculates the, hopeful, increase in the score of the Domsize
  

10) Week of 14/11:

    Thinking of using the Bayesian Optimization Method. 
    
    Indeed, it replies perfectly to my task which is to choose the best parameters to reach the highest score
    
10) Week of 21/11:

    Added the main algorithm notebook, that is a cleaned notebook that shows all the steps and why I did it
    
    It also contains a proof of the application by creating the new dataset
    
11) Week of 28/11:

  - Cleaned the Notebook
  - Removed the old notebooks
  - Tried other design of ML methods
  - Created other design of ML algorithm: Moved to regression task instead of classification


12) Week of 5/12:

 - Final report


13) Week of 12/12:


    - Final report
