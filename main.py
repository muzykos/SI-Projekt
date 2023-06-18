from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pandas as pd
import csv
import nltk
nltk.download('wordnet') 
from nltk.stem import WordNetLemmatizer
import re
from wordProcesser import preprocess

def parsePolarity(polarity):
    if not polarity:
        return 'negative'
    elif polarity == 2:
        return 'positive'
    return -1

# Write here what % of samples should be in test_data   
test_part = 0.2
# example_limit = 1200

sample_count = 1600000
test_count = int(sample_count * test_part)

train_data = []
train_labels = []
test_data = []
test_labels = []

p = int(sample_count / test_count)

print("Reading twitter_emotions.csv...")
with open('twitter_emotions.csv', newline='', encoding="ISO-8859-1") as f:
    reader = csv.reader(f)
    for ir, row in enumerate(reader):
        data = []
        for ic, column in enumerate(row):
            if ic in [0, 5]:
                if not parsePolarity:
                    continue
                if not ic:
                    parsed = parsePolarity(int(column))
                    if(parsed):
                        data.append(parsed)
                else:
                    data.append(preprocess(column))
                    data = data[::-1]
                    if(ir % p == 0):
                        test_data.append(data[0])
                        test_labels.append(data[1])
                    else:
                        train_data.append(data[0])
                        train_labels.append(data[1])
                    continue
        if ir % 10000 == 0:
            print((ir*100)/1600000,"% complete...")
print("Reading complete.") 

print("Total number of samples: ", sample_count, ".\nWanted number of tests: ", test_count, 
      ".\nActual number of tests: ", len(test_data), ".\nNumber of train samples: ", len(train_data),".")

# Vectorizing text data
vectorizer = TfidfVectorizer()
train_features = vectorizer.fit_transform(train_data)
test_features = vectorizer.transform(test_data)

# Training the classifier
classifier = MultinomialNB()
classifier.fit(train_features, train_labels)

# Predicting test samples
predictions = classifier.predict(test_features)

# Print the predictions (uncommenting would make it take moore time)
for text, prediction in zip(test_data, predictions):
    print(f"Text: {text} | Sentiment: {prediction}")

# Evaluating the accuracy of the classifier
accuracy = accuracy_score(test_labels, predictions)
print(f"Accuracy: {accuracy}")