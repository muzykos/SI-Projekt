import pandas as pd

from pyarc import TransactionDB
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
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
pd.__version__

clf = DecisionTreeClassifier(random_state=0,min_samples_split=10)
le = preprocessing.LabelEncoder()

# # Load the red wine dataset
# red_wine_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
# red_wine_data = pd.read_csv(red_wine_url, delimiter=';')


# Load the white wine dataset
white_wine_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
white_wine_data = pd.read_csv(white_wine_url, delimiter=';')


# Separate the features (X) and the target labels (y) for each dataset
red_wine_X = white_wine_data.drop('quality', axis=1)
red_wine_y = white_wine_data['quality']

"""
white_wine_X = white_wine_data.drop('quality', axis=1)
white_wine_y = white_wine_data['quality']
"""

# Split the dataset into training and testing sets
#X_train, X_test, y_train, y_test = train_test_split(red_wine_X,red_wine_y, test_size=0.2, random_state=42)
#training_data, testing_data = train_test_split(red_wine_data, test_size=0.2, random_state=42)
training_data, testing_data = train_test_split(white_wine_data, test_size=0.2)

# Convert the dataset to TransactionDB format
Train_txns = TransactionDB.from_DataFrame(training_data)
Test_txns = TransactionDB.from_DataFrame(testing_data)

#for i in Train_txns: print(i)

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

# clf.fit(red_wine_X,red_wine_y)

# p = clf.predict(red_wine_X)

# #fig = plt.figure(figsize=(25,20))
# fig = plt.figure(figsize=(100,80))
# _ = tree.plot_tree(clf, fontsize=10,
#                    filled=True)

# fig.savefig("decistion_tree.png")
