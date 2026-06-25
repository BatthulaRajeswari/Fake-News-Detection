import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier

fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

fake["label"] = "FAKE"
true["label"] = "REAL"

data = pd.concat([fake, true])

X = data["text"]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

tfidf = TfidfVectorizer(stop_words="english", max_df=0.7)

X_train_tfidf = tfidf.fit_transform(X_train)

model = PassiveAggressiveClassifier(max_iter=50)
model.fit(X_train_tfidf, y_train)

joblib.dump(model, "model.pkl")
joblib.dump(tfidf, "vectorizer.pkl")

print("Files created successfully!")
