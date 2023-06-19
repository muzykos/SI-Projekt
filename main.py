from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import csv
from wordProcesser import preprocess
import time

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
step = 10000


print(test_part)
train_data = []
train_labels = []
test_data = []
test_labels = []

p = int(sample_count / test_count)
print(p)

t=time.time()

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
                    if parsed:
                        data.append(parsed)
                    else:
                        continue
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
        if ir % step == 0:
            percent = (ir*100)/1600000
            print(f"{percent}% completed...")

print("Reading completed.") 

print(f"BENCHMARK - read+preprocess {time.time()-t}")

act_tests = len(test_data)
act_trains = len(train_data)

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
classifier.fit(train_features, train_labels)
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
accuracy = accuracy_score(test_labels, predictions)
print(f"Classifier accuracy: {accuracy}")
print(f"BENCHMARK - everything {time.time()-t}")