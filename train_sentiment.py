from app.task import clean_text
import pickle
import random
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# ------ Small Dataset -------

positive_samples = [
    "I had a productive day and accomplished a lot",
    "Feeling great today, very motivated!",
    "I solved three problems and I'm happy",
    "This was a very good session, confident and energetic",
    "I did well in the mock interview, so relieved"
]

negative_samples = [
    "I am stressed and can't focus on coding",
    "Today was terrible, I failed so many attempts",
    "Feeling anxious about placements and interviews",
    "I feel demotivated and tired, nothing went right",
    "Very bad day, no progress on problems"
]

data = []
labels = []

for s in positive_samples:
    for i in range(6):
        data.append(clean_text(s + ("!" * (i % 3))))
        labels.append(1)
for s in negative_samples:
    for i in range(6):
        data.append(clean_text(s + ("." * (i % 3))))
        labels.append(0)

combined = list(zip(data, labels))
random.shuffle(combined)
data, labels = zip(*combined)

# ---- Train / Test split -----
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

#---- Vectorizer + Classifier ------
vec = TfidfVectorizer(ngram_range=(1,2), max_features=5000)
clf = LogisticRegression(solver="liblinear", max_iter=500)

# Fit
X_train_vec = vec.fit_transform(X_train)
clf.fit(X_train_vec, y_train)

#Evaluate
X_test_vec = vec.transform(X_test)
y_pred = clf.predict(X_test_vec)
acc = accuracy_score(y_test, y_pred)

print("Accuracy on demo test set:", acc)
print("Classification report:")
print(classification_report(y_test, y_pred, target_names=["negative", "positive"]))

# ------- Save model + vectorizer -------
os.makedirs("models", exist_ok=True)
with open("models/vectorizer.pkl", "wb") as f:
    pickle.dump(vec, f)
with open("models/sentiment_model.pkl", "wb") as f:
    pickle.dump(clf, f)

print("\nSaved vectorizer -> models/vectorizer.pkl")
print("\nSaved model -> models/sentiment_model.pkl")

# ------- Helper predict function ------
def predict_sentiment_demo(text: str):
    """
    Loads model files and returns (label, prob_positive)
    label: 'positive' or 'negative'
    """
    t = clean_text(text)
    with open("models/vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open("models/sentiment_model.pkl", "rb") as f:
        model = pickle.load(f)
    
    X = vectorizer.transform([t])
    proba = model.predict_proba(X)[0]
    label = "positive" if proba[1] >= 0.5 else "negative"
    return label, float(proba[1])

if __name__ == "__main__":
    samples = [
        "I feel great and productive today",
        "Can't focus and I'm anxious about interviews"
    ]
    print("\nSanity checks:")
    for s in samples:
        lbl, p = predict_sentiment_demo(s)
        print(f"  '{s}' -> {lbl} (positive_prob={p:.2f})")