import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import json


from prepare_text import prepare_text, delete_empty, filter_words
from make_dictionary import make_dictionary
from cut_non_toxic import cut_non_toxic


# prepare data
columns_to_analyze = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
data = pd.read_csv("data/train.csv")
data['cleaned_comment_text'] = data['comment_text'].apply(prepare_text)
data = delete_empty(data)

key_words = make_dictionary(data, columns_to_analyze)
with open('data/key_words.json', 'w') as file:
    json.dump(key_words, file)
data['cleaned_comment_text'] = data['cleaned_comment_text'].apply(lambda x: filter_words(x, key_words))

data = cut_non_toxic(data, 105000, columns_to_analyze)

data.to_csv("data/prepared_train.csv")

# vectorize
X_train, X_test, y_train, y_test = train_test_split(data['cleaned_comment_text'], data[columns_to_analyze], test_size=0.2, random_state=42)

tfidf_vectorizer = TfidfVectorizer(max_features=500)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
joblib.dump(tfidf_vectorizer, 'models/tfidf_vectorizer.joblib')
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# models
for label in columns_to_analyze:
    model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    model.fit(X_train_tfidf, y_train[label])

    joblib.dump(model, f'models/{label}.joblib')
