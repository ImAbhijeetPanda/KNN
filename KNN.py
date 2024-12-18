                    #K-Nearest Neighbors#

'''
Problem Statement:

1.	A glass manufacturing plant uses different earth elements to design new glass materials based on customer requirements. For that, they would like to automate the process of classification as it’s a tedious job to manually classify them. Help the company achieve its objective by correctly classifying the glass type based on the other features using KNN algorithm. 
'''

'''
RI: Refractive Index
Na: Sodium
Mg: Magnesium
Al: Aluminum
Si: Silicon
K: Potassium
Ca: Calcium
Ba: Barium
Fe: Iron
Type: Glass type (1, 2, 3, etc.)
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# MySQL Database connection

from sqlalchemy import create_engine, text

glass = pd.read_csv(r"A:/360DigiTM/ML/Assignment/KNN/glass.csv")

# Creating engine which connect to MySQL
user = 'user1' # user name
pw = 'user1' # password
db = 'classb' # database

# creating engine to connect database
engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")

# dumping data into database 
glass.to_sql('glass', con = engine, if_exists = 'replace', chunksize = 1000, index = False)



# loading data from database
sql = 'select * from glass'

glass_df = pd.read_sql_query(text(sql), con = engine.connect())

print(glass_df)

df = glass_df.copy()

df.nunique()
sum(df.duplicated())

df = df.drop_duplicates()

df.info()


df = df.reset_index(drop = True)


# Split the data into features and target
x = df.drop('Type', axis=1)
y = df['Type']

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

X_train.shape
X_test.shape


# Train a Random Forest Classifier
from sklearn.neighbors import KNeighborsClassifier
y_test.nunique()
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, y_train)


# Predict on the test set
pred_train = knn.predict(X_train)


# Evaluate the model with train data
# Cross table
pd.crosstab(y_train, pred_train, rownames = ['Actual'], colnames = ['Predictions']) 
from sklearn.metrics import accuracy_score
 # Accuracy measure
accuracy_train = accuracy_score(y_train, pred_train)
print(f'Accuracy train: {accuracy_train*100}%')

import sklearn.metrics as skmet
cm1 = skmet.confusion_matrix(y_train, pred_train)

cmplot1 = skmet.ConfusionMatrixDisplay(confusion_matrix = cm1, display_labels = ['1', '2','3','5','6','7'])

cmplot1.plot()
cmplot1.ax_.set(title = 'Glass Type Detection - Confusion Matrix', 
               xlabel = 'Predicted Value', ylabel = 'Actual Value')


# Predict the class on test data
pred = knn.predict(X_test)
pred
# Cross table
pd.crosstab(y_test, pred, rownames = ['Actual'], colnames = ['Predictions']) 


# Evaluate the model with test data
from sklearn.metrics import accuracy_score

accuracy_test = accuracy_score(y_test, pred)
print(f'Accuracy test: {accuracy_test*100}%')


import sklearn.metrics as skmet
cm = skmet.confusion_matrix(y_test, pred)

cmplot = skmet.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = ['1', '2','3','5','6','7'])

cmplot.plot()
cmplot.ax_.set(title = 'Glass Type Detection - Confusion Matrix', 
               xlabel = 'Predicted Value', ylabel = 'Actual Value')


# creating empty list variable 
acc = []

# running KNN algorithm for 3 to 50 nearest neighbours(odd numbers) and 
# storing the accuracy values

for i in range(1, 100,3):
    neigh = KNeighborsClassifier(n_neighbors = i)
    neigh.fit(X_train, y_train)
    train_acc = np.mean(neigh.predict(X_train) == y_train)
    test_acc = np.mean(neigh.predict(X_test) == y_test)
    diff = train_acc - test_acc
    acc.append([diff, train_acc, test_acc])
    
acc
    
# Plotting the data accuracies
plt.figure(figsize=(20, 10))
plt.plot(np.arange(1, 100,3), [i[1] for i in acc], "r-")
plt.plot(np.arange(1, 100,3), [i[2] for i in acc], "b-")
plt.show()

# Hyperparameter optimization
from sklearn.model_selection import GridSearchCV 
#Optimization algorithim which gives the best parameters based on the scoring mariic

k_range = list(range(1, 50))
param_grid = dict(n_neighbors = k_range)
  
# Defining parameter range
grid = GridSearchCV(knn, param_grid, cv = 6, 
                    scoring = 'accuracy', 
                    return_train_score = False, verbose = 1)

KNN_new = grid.fit(X_train, y_train)


print("Best parameters found: ", KNN_new.best_params_)
accuracy = KNN_new.best_score_ *100
print("Accuracy for our training dataset with tuning is : {:.2f}%".format(accuracy) )



# Predict the class on test data
pred = KNN_new.predict(X_test)
pred

cm = skmet.confusion_matrix(y_test, pred)

cmplot = skmet.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = ['Benign', 'Malignant'])
cmplot.plot()
cmplot.ax_.set(title = 'Cancer Detection - Confusion Matrix', 
               xlabel = 'Predicted Value', ylabel = 'Actual Value')

skmet.accuracy_score(y_test,pred)*100

# Predict the class on train data
pred_tr = KNN_new.predict(X_train)
pred_tr

cm = skmet.confusion_matrix(y_train, pred_train)

cmplot = skmet.ConfusionMatrixDisplay(confusion_matrix = cm)
cmplot.plot()
cmplot.ax_.set(title = 'Glass Type Detection - Confusion Matrix', 
               xlabel = 'Predicted Value', ylabel = 'Actual Value')

skmet.accuracy_score(y_train,pred_tr)*100

# Save the model
import pickle
knn_best = KNN_new.best_estimator_
pickle.dump(knn_best, open('knn.pkl', 'wb'))

import os
os.getcwd()































