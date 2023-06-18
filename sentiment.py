import nltk

from sklearn import datasets
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from frlearn.base import probabilities_from_scores, select_class
from frlearn.classifiers import FRNN
from frlearn.feature_preprocessors import RangeNormaliser

#data
text_data = []

processed_data = []
for text, label in text_data:
    text = text.lower()



# Import example data.
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split into train and test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)

# Create an instance of the FRNN classifier, construct the model, and query on the test set.
clf = FRNN(preprocessors=(RangeNormaliser(), ))
model = clf(X_train, y_train)
scores = model(X_test)

# Convert scores to probabilities and calculate the AUROC.
probabilities = probabilities_from_scores(scores)
auroc = roc_auc_score(y_test, probabilities, multi_class='ovo')
print('AUROC:', auroc)

# Select classes with the highest scores and calculate the accuracy.
classes = select_class(scores)
accuracy = accuracy_score(y_test, classes)
print('accuracy:', accuracy)