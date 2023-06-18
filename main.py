import pandas as pd

import numpy as np

from itertools import combinations
from pyarc import TransactionDB
from pyarc import CBA
from pyarc.data_structures import Item
from pyarc.algorithms import (
    top_rules,
    createCARs,
    M1Algorithm,
    M2Algorithm
)

from matplotlib import pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
#from roughsets_base.roughset_dt import RoughSetDT
from roughsets_base.roughset_dt import RoughSetDT
from sklearn.feature_selection import RFECV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
pd.__version__



clf = DecisionTreeClassifier(random_state=0,min_samples_split=10)
le = preprocessing.LabelEncoder()

def loaddataset(url):
    return pd.read_csv(url, delimiter=';')

def cleanupdataset(dataset):
    df = pd.DataFrame(dataset)
    df.fillna(df.median(numeric_only=True).round(1), inplace=True)
    return df

def feature_selection(dataset):
    # Separate the features (X) and the target labels (y) for each dataset
    X = dataset.drop('quality', axis=1)
    y = dataset['quality']

    # Create a classifier for feature selection
    classifier = RandomForestClassifier(n_estimators=100)

    # Create a cross-validation strategy
    cv = StratifiedKFold(n_splits=5)

    # Create the RFE selector with cross-validation
    rfe = RFECV(estimator=classifier, cv=cv)

    # Perform feature selection
    rfe.fit(X, y)

    # Get the selected features
    selected_features = rfe.support_
    
    #selected_columns = X.columns[selected_features]
    #print("Selected columns:")
    #print(selected_columns)
    
    return selected_features

def reduce_dataset(dataset,selected_features):
    # Separate the features and the target variable
    X = dataset.drop('quality', axis=1)
    y = dataset['quality']

    # Remove non-selected features from the dataset
    X_selected = X.loc[:, selected_features]

    selected_data = pd.concat([X_selected, y], axis=1)

    # Print the selected features
    # print("reduced dataset:")
    # print(selected_data)

    return selected_data



def train_model(dataset):
    #RoughSetDT.__reduce__
    # Split the dataset into training and testing sets

    training_data, testing_data = train_test_split(dataset, test_size=0.2)

    # Convert the dataset to TransactionDB format
    Train_txns = TransactionDB.from_DataFrame(training_data)
    Test_txns = TransactionDB.from_DataFrame(testing_data)

    
    # Train the Rough Sets model
    rules = top_rules(Train_txns.string_representation)

    # Print the generated rules
    # for rule in rules:
    #      print(rule)

    cars = createCARs(rules)


    classifier = M1Algorithm(cars, Train_txns).build()
    # classifier = M2Algorithm(cars, txns_train).build()


    accuracy = classifier.test_transactions(Test_txns)
    print(accuracy)
    return classifier

def predict(classifier,dataset):
    input_data = pd.DataFrame({
    'fixed acidity': [7.2],
    'volatile acidity': [0.36],
    'citric acid': [0.46],
    'residual sugar': [2.1],
    'chlorides': [0.074],
    'free sulfur dioxide': [12],
    'total sulfur dioxide': [87],
    'density': [0.997],
    'pH': [3.35],
    'sulphates': [0.48],
    'alcohol': [10.2],
    })
    predictions = classifier.predict(input_data)

    print(predictions)

def main():
    # # Load the red wine dataset
    # red_wine_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    # red_wine_data = pd.read_csv(red_wine_url, delimiter=';')


    # Load the white wine dataset
    white_wine_data = loaddataset("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv")
    white_wine_data = cleanupdataset(white_wine_data)

    features = feature_selection(white_wine_data)
    #print(features)
    dataset = reduce_dataset(white_wine_data,features)
    print(dataset)
    #print(dataset.iloc[1])
    classifier = train_model(dataset)
    #print(classifier)
    #predict(classifier,dataset)
    





if __name__ == "__main__":
    main()



