import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet
import contractions

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4')
try:
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
except LookupError:
    nltk.download('averaged_perceptron_tagger_eng')

lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    return wordnet.NOUN

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = contractions.fix(text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    lemmatized = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in tagged]
    return ' '.join(lemmatized)

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

train["text"] = (train["Title"].fillna('') + " " + train["Description"].fillna('')).apply(clean_text)
test["text"] = (test["Title"].fillna('') + " " + test["Description"].fillna('')).apply(clean_text)

X_train = train["text"]
y_train = train["Class Index"]
X_test = test["text"]
y_test = test["Class Index"]

model_configs = {
    "Logistic Regression": {
        "model": LogisticRegression(max_iter=1000),
        "params": {
            'tfidf__ngram_range': [(1, 1), (1, 2)],
            'tfidf__max_df': [0.7, 1.0],
            'clf__C': [0.1, 1, 10]
        }
    },
    "Multinomial Naive Bayes": {
        "model": MultinomialNB(),
        "params": {
            'tfidf__ngram_range': [(1, 1), (1, 2)],
            'tfidf__max_df': [0.7, 1.0],
            'clf__alpha': [0.5, 1.0, 2.0]
        }
    }
}

for name, config in model_configs.items():
    print(f"\n==== {name} ====")
    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('clf', config["model"])
    ])
    grid = GridSearchCV(pipe, config["params"], cv=3, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)
    y_pred = grid.predict(X_test)

    print("Best Parameters:", grid.best_params_)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
