import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from fuzzy_rough_learn import FuzzyRoughClassifier
from fuzzy_rough_learn.reducts import get_reducts

# Step 1: Data Preprocessing
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Example text data
text_data = [
    ("I love this movie", "positive"),
    ("This book is boring", "negative"),
    ("Great experience at the restaurant", "positive"),
    ("The service was terrible", "negative")
]

preprocessed_data = []
for text, label in text_data:
    text = text.lower()
    tokens = word_tokenize(text)
    preprocessed_text = [
        lemmatizer.lemmatize(word) for word in tokens if word not in stop_words
    ]
    preprocessed_data.append((" ".join(preprocessed_text), label))

# Step 2: Feature Extraction
vectorizer = CountVectorizer()
X = vectorizer.fit_transform([data[0] for data in preprocessed_data])
y = np.array([data[1] for data in preprocessed_data])

# Step 3: Rough Set Theory with fuzzy-rough-learn
X = X.toarray()  # Convert the sparse matrix X to a dense array

reducts = get_reducts(X, y)

classifier = FuzzyRoughClassifier()
classifier.set_reducts(reducts)
classifier.fit(X, y)

# Step 4: Classification
# Make predictions and evaluate the performance
preprocessed_test_data = [
    ("I really enjoyed the movie", "positive"),
    ("The book was not interesting", "negative")
]

X_test = vectorizer.transform([data[0] for data in preprocessed_test_data]).toarray()
y_test = np.array([data[1] for data in preprocessed_test_data])

predictions = classifier.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)