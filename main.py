from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from samples import samples
import csv

def parsePolarity(polarity):
    if not polarity:
        return 'negative'
    elif polarity == 2:
        return 'positive'
    return -1

# Write here what % of samples should be in test_data   
test_part = 0.1
# example_limit = 1200

sample_count = 1600000
test_count = int(sample_count * test_part)

train_data = []
test_data = []
test_labels = []

p = int(sample_count / test_count)

with open('twitter_emotions.csv', newline='') as f:
    reader = csv.reader(f)
    for ir, row in enumerate(reader):
        data = []
        print(row)
        for ic, column in enumerate(row):
            if ic in [0, 5]:
                # print(column, ic, end=',')
                if not parsePolarity:
                    continue
                if not ic:
                    parsed = parsePolarity(int(column))
                    if(parsed):
                        data.append(parsed)
                else:
                    data.append(column)
                    data = data[::-1]
            if len(data) == 2:
                if(ir % p == 0):
                    test_data.append(data[0])
                    test_labels.append(data[1])
                else:
                    train_data.append(data)

# sample_count = len(samples)
# test_count = int(sample_count * test_part)

# p = int(sample_count / test_count)

# for id, sample in enumerate(samples):
#     if id % p == 0:
#         test_data.append(sample[0])
#         test_labels.append(sample[1])
#     else:
#         train_data.append(sample)

print(sample_count, test_count, len(test_data), len(train_data))

# Preprocess the training data
train_text = [data[0] for data in train_data]
train_labels = [data[1] for data in train_data]

# Preprocess the test data
test_text = test_data

# Vectorize the text data
vectorizer = TfidfVectorizer()
train_features = vectorizer.fit_transform(train_text)
test_features = vectorizer.transform(test_text)

# Train the classifier
classifier = MultinomialNB()
classifier.fit(train_features, train_labels)

# Make predictions on the test data
predictions = classifier.predict(test_features)

# Print the predictions
# for text, prediction in zip(test_text, predictions):
#     print(f"Text: {text} | Sentiment: {prediction}")

# Evaluate the accuracy of the classifier
accuracy = accuracy_score(test_labels, predictions)
print(f"Accuracy: {accuracy}")