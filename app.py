import pandas as pd;
df=pd.read_csv("training.csv");
#preview it
print(df.head());
print(df.info())
print(df['label'].value_counts())
label_map = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear"
}

df['mood'] = df['label'].map(label_map)
print(df[['text', 'label', 'mood']].head(10))
print(df['mood'].value_counts())

import matplotlib.pyplot as plt

df['mood'].value_counts().plot(kind='bar', color=['blue', 'gold', 'pink', 'red', 'purple'])
plt.title('Mood Frequency')
plt.xlabel('Mood')
plt.ylabel('Count')
plt.tight_layout()
plt.show()
import re

def clean_text(text):
    text = text.lower()                           # Convert text to lowercase
    text = re.sub(r'http\S+', '', text)           # Remove URLs
    text = re.sub(r'@\w+', '', text)              # Remove mentions
    text = re.sub(r'#\w+', '', text)              # Remove hashtags
    text = re.sub(r'[^\w\s]', '', text)           # Remove punctuation
    text = re.sub(r'\d+', '', text)               # Remove numbers
    text = re.sub(r'\s+', ' ', text).strip()      # Whitespace cleanup
    return text

df['clean_text'] = df['text'].apply(clean_text)
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer=TfidfVectorizer(max_features=5000)
X=vectorizer.fit_transform(df['clean_text'])
from sklearn.preprocessing import LabelEncoder
Label_encoder=LabelEncoder()
Y=Label_encoder.fit_transform(df['mood'])
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
from sklearn.linear_model import LogisticRegression
clf=LogisticRegression(max_iter=1000)
clf.fit(X_train,y_train)
from sklearn.metrics import accuracy_score, classification_report

y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
sample = ["I am so angry at you"]
vec = vectorizer.transform(sample)
pred = clf.predict(vec)
print(Label_encoder.inverse_transform(pred))
import joblib

# Save the TF-IDF vectorizer
joblib.dump(vectorizer, "vectorizer.pkl")

# Save the trained classifier
joblib.dump(clf, "model.pkl")

# Save the label encoder
joblib.dump(Label_encoder, "label_encoder.pkl")