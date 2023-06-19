from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split #do opisania
import csv
from wordProcesser import preprocess
import random #do opisania
import time #do wywalenia

def parsePolarity(polarity):
    if not polarity:
        return 'negative'
    elif polarity == 4:
        return 'positive'
    return -1

# What % of samples should be in test_data   
test_part = 5 # %

# Parameters to set for TF-IDF vectorizer


# Parameters to set for Naive Bayes Multinomial classifier
nb_alpha = 1.0 #1.0 default
nb_force_alpha = True #True default
nb_fit_prior = True #True default

test_part *= 0.01
sample_count = 1600000
test_count = int(sample_count * test_part)
step = 40000

read_data = []
read_label = []
train_data = []
train_label = []
test_data = []
test_label = []

p = int(sample_count / test_count)

t=time.time()
print("Reading twitter_emotions.csv...")
with open('twitter_emotions.csv', newline='', encoding="ISO-8859-1") as f:
    reader = csv.reader(f)
    for ir, row in enumerate(reader):
        data = []
        for ic, column in enumerate(row):
            if ic in [0, 5]:
                if not ic:
                    parsed = parsePolarity(int(column))
                    if not parsed:
                        continue
                    read_label.append(parsed)
                else:
                    read_data.append(column)
                    continue
        if ir % step == 0:
            percent = (ir*100)/1600000
            print(f"{percent}% completed...")
print("Shuffling...")
random.shuffle(read_data)
print("Reading completed.")

print("Preprocessing read words...")
preprocess(read_data)
print("Preprocessing completed.")
print(f"BENCHMARK - read+process {time.time()-t}")

print("Splitting into train and test groups...")
train_data, test_data, train_label, test_label = train_test_split(read_data, read_label,
                                                                   test_size=test_part)
act_tests = len(test_data)
act_trains = len(train_data)
print("Splitting completed.")


print(f"Total number of samples: {sample_count}.\nWanted number of tests: {test_count}.\nActual number of tests: {act_tests}.\nNumber of train samples: {act_trains}.")

print("Vectorizing data...")
# Vectorizing text data, sadly % completed is impossible without modifying library
vectorizer = TfidfVectorizer(lowercase=False) # already lowercased
train_features = vectorizer.fit_transform(train_data)
test_features = vectorizer.transform(test_data)
print("Vectorizing completed.")

# Training the classifier, sadly % completed is impossible without modifying library
classifier = MultinomialNB(alpha=nb_alpha, force_alpha=nb_force_alpha, fit_prior=nb_fit_prior)

print("Training classifier...")
# step_train = step * (1-test_part)
classifier.fit(train_features, train_label)
print("Training completed.")

# Predicting test samples, % completed is slowing it down a lot
predictions = []

print("Predicting test samples...")
# step_test = step * test_part
# for i in range(0, act_tests):
    # prediction = classifier.predict(test_features[i])
    # predictions.append(prediction)
    # if i % step_test == 0:
        # percent = (i*100)/act_tests
        # print(f"{percent}% completed...")
predictions = classifier.predict(test_features)
print("Predicting completed.")

# Print the predictions (uncommenting would make it take mooore time)
# for text, prediction in zip(test_data, predictions):
    # print(f"Text: {text} | Sentiment: {prediction}")

# Evaluating the accuracy of the classifier
accuracy = accuracy_score(test_label, predictions)
print(f"Classifier accuracy: {accuracy}")
print(f"BENCHMARK - everything: {time.time()-t}")