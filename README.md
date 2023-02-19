# Sampling

Here we will perform various Sampling techniques on our Creditcard dataset. We will explore the imbalanced dataset issues and 
implementation of various techniques on different ML models. We will further analyze their accuracies and discuss the results.

## Methodolgy
### 1. Converting Imbalanced dataset to balanced dataset: <br>
   In the given dataset, the class '1' has less number of samples. We solved this issue by oversampling class '1' instances
   and making them equal to class '0' instances. Saving the new balanced dataset into "Balanced.csv".
   
### 2. Generating samples using four different sampling techniques: <br>
   Exploring various sampling techniques:
   1. **Simple Random Sampling** : A simple random sample is a subset of individuals chosen from a larger set in which a subset of individuals are chosen randomly, all       with the same probability. We found the sample size using Sample size detection formula :

      <img src="https://user-images.githubusercontent.com/100415671/219952601-de539e8a-3dad-4da7-8722-e8304a1c5396.png" width=30% height=30%>

   

   2. **Systematic Sampling** : Systematic sampling is a probability sampling method where researchers select members of the population at a regular interval. We             defined the step size by passing the arguments. 
      
   3. **Stratified Sampling** : Stratified sampling is a method of sampling from a population which can be partitioned into subpopulations. Here it is based on the           class attribute. Then we select equal number of instances of both the subpopulations.
      
   4. **Cluster Sampling** : A probability sampling method in which you divide a population into clusters   and then randomly select some of these clusters as your            sample.  
   5. **Convenience Sampling** :Convenience sampling is a non-probability sampling method where units are selected for inclusion in the sample because they are the            easiest for the researcher to access.
### 3. Applying five ML Models on above four samples: <br>

   - **Random Forest**
   - **Decision Tree**
   - **K-Nearest Neighbour**
   - **Logisitic Regression**
   - **XGBoost**

### 4. Constructing Accuracy Table : <br>
   <img src="https://user-images.githubusercontent.com/100415671/219959246-b0224ef3-e002-40a3-b8c9-98bc1bc9c74a.png" width=60% height=70%>
   
   
## Conclusion
 As there are mutliple 100s in the table, it is unable to say one model. It depends on the sample taken using the technique. But the model which gives best result 
 with every technique is **XGBoost**





     

   
   







