# Sampling

Here we will perform various Sampling techniques on our Creditcard dataset. We will explore the imbalanced dataset issues and 
implementation of various techniques on different ML models. We will further analyze their accuracies and discuss the results.

## Methodolgy
### 1. Converting Imbalanced dataset to balanced dataset: <br>
   In the given dataset, the class '1' has less number of samples. We solved this issue by oversampling class '1' instances
   and making them equal to class '0' instances. Saving the new balanced dataset into "Balanced.csv".
   
### 2. Generating samples using four different sampling techniques: <br>
   Exploring various sampling techniques:
   1. Simple Random Sampling :A simple random sample is a subset of individuals chosen from a larger set in which a subset of individuals are chosen randomly, all with         the same probability. We found the sample size using Sample size detection formula :

      <img src="https://user-images.githubusercontent.com/100415671/219952601-de539e8a-3dad-4da7-8722-e8304a1c5396.png" width=50% height=50%>





