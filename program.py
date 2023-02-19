import pandas as pd
import random
import numpy as np
from sklearn import linear_model
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit

#Reading the csv file
data=pd.read_csv("Creditcard_data.csv")

#print(data.shape)
class_count_0, class_count_1 = data['Class'].value_counts()

# # Separate class
class_0 = data[data['Class'] == 0]
class_1 = data[data['Class'] == 1]# print the shape of the class
# print('class 0:', class_0.shape)
# print('class 1:', class_1.shape)


# Oversampling : since class 1 has less number of instances.
class_1_over = class_1.sample(class_count_0, replace=True)

test_over = pd.concat([class_1_over, class_0], axis=0)

test_over.to_csv("Balanced.csv",index=False)
data1=pd.read_csv("Balanced.csv")


def systematic_sampling(df, step):
 
    indexes = np.arange(0, len(df), step=step)
    systematic_sample = df.iloc[indexes]
    return systematic_sample


def cluster_sampling(df, number_of_clusters):

    try:
        # Divide the units into cluster of equal size
        df['cluster_id'] = np.repeat(
            [range(1, number_of_clusters+1)], len(df)/number_of_clusters)

        # Create an empty list
        indexes = []

        # Append the indexes from the clusters that meet the criteria
        # For this formula, clusters id must be an even number
        for i in range(0, len(df)):
            if df['cluster_id'].iloc[i] % 3 == 0:
                indexes.append(i)
        cluster_sample = df.iloc[indexes]
        return (cluster_sample)

    except:
        print("The population cannot be divided into clusters of equal size!")


# Simple Random Sampling

Z=1.96  # Z-score corresponding to desired level of confidence (1.96 for 95% confidence)
p=0.5  # estimated proportion of population with a certain characteristic (assumed to be 0.5 )
E=0.05  #desired margin of error

# Sample-size detection formula
num1 = (pow(Z,2)*p*(1-p))/pow(E,2)
num1 = round(num1)

data1= shuffle(data1)
s1 = data1.sample(n=num1)

# Systematic Sampling
s2 = systematic_sampling(data1, 6)

# Cluster Sampling
s3= cluster_sampling(data1,7)

# Stratified Sampling
s4= data1.groupby('Class', group_keys=False).apply(lambda x: x.sample(300)) 

#Convenience Sampling

s5=data1.head(400)
# Creation of dataframe
accuracy=pd.DataFrame()

# Creating the sample list
sample=[]
sample.append(s1)
sample.append(s2)
sample.append(s3)
sample.append(s4)
sample.append(s5)

# Declaring names of Sampling Techniques

sample_names=["Simple Random","Systematic","Cluster","Stratified","Convenience"]

 
# ML Models
for i in range(0,5):

    acc=[]
    X= sample[i].drop(sample[i].columns[-1],axis=1)
    y=sample[i].iloc[:,30]
    X_train, X_test,y_train, y_test = train_test_split(X,y ,
                                    random_state=1044, 
                                    test_size=0.2, 
                                    shuffle=True)

    logr = linear_model.LogisticRegression(max_iter=5000)
    logr.fit(X_train,y_train)
    y_pred_log= logr.predict(X_test)
    score_log_1=accuracy_score(y_test,y_pred_log)
    acc.append(score_log_1*100)

    #KNN
    knn = KNeighborsClassifier(n_neighbors = 10)
    knn.fit(X_train, y_train)
    y_pred_knn= knn.predict(X_test)
    score_knn_1=accuracy_score(y_test,y_pred_knn)
    acc.append(score_knn_1*100)

    #DECISION TREE
    model = tree.DecisionTreeClassifier()
    model = model.fit(X_train, y_train)
    y_pred_tree = model.predict(X_test)
    score_tree_1=accuracy_score(y_test,y_pred_tree)
    acc.append(score_tree_1*100)

    #RANDOM FOREST
    clf = RandomForestClassifier(n_estimators =5)
    clf.fit(X_train, y_train)
    y_pred_rf = clf.predict(X_test)
    score_rf_1=accuracy_score(y_test,y_pred_rf)
    acc.append(score_rf_1*100)

    #XGB
    my_model = xgb.XGBClassifier()
    my_model.fit(X_train, y_train)
    y_pred_xgb= my_model.predict(X_test)
    score_xgb_1=accuracy_score(y_test,y_pred_xgb)
    acc.append(score_xgb_1*100)
    name=sample_names[i]
    accuracy[f"{name}"]=acc

accuracy.index = ['Logisitic', 'KNN', 'Decision Tree', 'Random Forest','XGBoost']

print(accuracy)
accuracy.to_csv("Final1.csv")