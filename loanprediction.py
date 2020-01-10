#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 13:51:20 2019

@author: suyash
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

pd.options.mode.chained_assignment = None

train = pd.read_csv('train_v2.csv')

train.fillna(train.mean(), inplace=True)
train.dropna(inplace=True)


correlations_data = train.corr()['loss'].sort_values()

for i in train.columns:
    if len(set(train[i]))==1:
        train.drop(labels=[i], axis=1, inplace=True)
        
correlations_data = train.corr()['loss'].sort_values()

def delete_unnecessary(x, threshold):
    y = x['loss']
    x = x.drop(columns = ['loss'])
    
    # correlation matrix
    corr_matrix = x.corr()
    iters = range(len(corr_matrix.columns) - 1)
    drop_cols = []

    # Iterate correlation matrix and compare correlations
    for i in iters:
        for j in range(i):
            item = corr_matrix.iloc[j:(j+1), (i+1):(i+2)]
            col = item.columns
            val = abs(item.values)
            
            
            if val >= threshold:
                drop_cols.append(col.values[0])

    drops = set(drop_cols)
    x = x.drop(columns = drops)
    
    # Add the score back in to the data
    x['loss'] = y
    return x

train = delete_unnecessary(train, 0.6);

features = train.drop(columns='loss')
target = pd.DataFrame(train['loss'])

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size = 0.2, random_state = 42)
      
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)      

# Convert y to one-dimensional vector
y_train = np.array(y_train).reshape((-1, ))
y_test = np.array(y_test).reshape((-1, ))   

# Function to error
def cross_val(X_train, y_train, model):
    # Applying k-Fold Cross Validation
    from sklearn.model_selection import cross_val_score
    accuracies = cross_val_score(estimator = model, X = X_train, y = y_train, cv = 5)
    return accuracies.mean()

# Takes in a model, trains the model, and evaluates the model on the test set
def fit_and_evaluate(model):    
    # Train the model
    model.fit(X_train, y_train)    
    # Make predictions and evalute
    model_pred = model.predict(X_test)
    model_cross = cross_val(X_train, y_train, model)    
    # Return the performance metric
    return model_cross

from sklearn.ensemble import RandomForestClassifier
random = RandomForestClassifier(n_estimators = 10, criterion = 'entropy')
random_cross = fit_and_evaluate(random)

print('Cross Validation Score = %0.4f' % random_cross)
