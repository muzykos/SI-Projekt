from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split #do opisania
import csv
from wordProcesser import preprocess
import random #do opisania
import time #do opisania

#todo: vectorizing przed split

def parsePolarity(polarity):
    if not polarity:
        return 'negative'
    elif polarity == 4:
        return 'positive'
    return -1

# Program settings
benchmark = True
print_predictions = False
ask_user = True
adv_settings = False
repeat = False # don't touch

# Parameters for train/test splitting  
test_part = 15 # % of samples used for tests
seed = 1 # None default, makes train_test_split give same output for certain seed and test_part


# Parameters to set for TF-IDF vectorizer
#none bruh

# Parameters to set for Naive Bayes Multinomial classifier
nb_alpha = 1.0 #1.0 default
nb_force_alpha = True #True default
nb_fit_prior = True #True default

# Initialization
sample_count = 1600000
step = 40000

if benchmark: t=time.time()

read_data = []; read_label = []
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
print("Reading completed.")

print("Preprocessing read words...")
preprocess(read_data)
print("Preprocessing completed.")

if benchmark: print(f"BENCHMARK - read+process {time.time()-t}")

while ask_user:
    if repeat:
        answ = input("Do you want to leave? (enter 'yes' to stop the program): ")
        if answ.lower().strip() == "yes": break
    if ask_user:
        test_part = int(input("% of samples to put in test group: "))
        test_part *= 0.01 # percent to float
        test_count = int(sample_count * test_part)

        seed = int(input("Seed for reploducible output (same seed and test %): "))
        answ = input("Print predictions? (yes/anything; will take more time): ")
        if answ.lower().strip() == "yes": print_predictions = True
        else: print_predictions = False
        answ = input("Advanced classifier parameters? (yes/anything): ")
        if answ.lower().strip() == "yes":
            adv_settings = True
        else:
            print("Then have fun!")
        
        if adv_settings:
            print("You are changing parameters of Naive Bayes Multinominal classifier.\n")
            print("Not giving an answer leaves default value.\n")
            nb_alpha = int(input("Alpha (default 1.0): "))
            answ = input("Force alpha (default: True; answer True or False): ")
            if answ.lower().strip() == "true": nb_force_alpha = True
            elif answ.lower().strip() == "": nb_force_alpha = True
            else: nb_force_alpha = False
            answ = input("Fit prior (default: True; answer True or False): ")
            if answ.lower().strip() == "true": nb_fit_prior = True
            elif answ.lower().strip() == "": nb_fit_prior = True
            else: nb_fit_prior = False

    if benchmark: t2 = time.time()
    print("Splitting into train and test groups...")
    train_data = []; test_data = []; train_label = []; test_label = []
    train_data, test_data, train_label, test_label = train_test_split(read_data, read_label,
                                                                    test_size=test_part,
                                                                    random_state=seed)
    act_tests = len(test_data)
    act_trains = len(train_data)
    print("Splitting completed.")

    print(f"Total number of samples: {sample_count}.\nNumber of tests: {act_tests}.\nNumber of train samples: {act_trains}.")

    print("Vectorizing data...")
    # Vectorizing text data, sadly % completed is impossible without modifying library
    vectorizer = TfidfVectorizer(lowercase=False) # already lowercased
    train_features = None; test_features = None
    train_features = vectorizer.fit_transform(train_data)
    test_features = vectorizer.transform(test_data)
    print("Vectorizing completed.")

    # Training the classifier, sadly % completed is impossible without modifying library
    classifier = None
    classifier = MultinomialNB(alpha=nb_alpha, force_alpha=nb_force_alpha, fit_prior=nb_fit_prior)

    print("Training classifier...")
    # step_train = step * (1-test_part)
    classifier.fit(train_features, train_label)
    print("Training completed.")

    # Predicting test samples, % completed is slowing it down a lot
    predictions = []

    print("Predicting test samples...")
    predictions = classifier.predict(test_features)
    print("Predicting completed.")

    # Print the predictions (uncommenting would make it take mooore time)
    if print_predictions:
        for text, prediction in zip(test_data, predictions):
            print(f"Text: {text} | Sentiment: {prediction}")

    # Evaluating the accuracy of the classifier
    accuracy = None
    accuracy = accuracy_score(test_label, predictions)
    print(f"Classifier accuracy: {accuracy}")
    if benchmark: print(f"BENCHMARK - splitting, vectorizing, training and testing: {time.time()-t2}")
    repeat = True