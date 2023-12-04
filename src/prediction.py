import pandas as pd
import json
import joblib
from prepare_text import prepare_text, filter_words


test = pd.read_csv('data/test.csv')
test['cleaned_comment_text'] = test['comment_text'].apply(prepare_text)
columns_to_analyze = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# filter words
key_words = []
with open('data/key_words.json', 'r') as file:
    key_words = json.load(file)
test['cleaned_comment_text'] = test['cleaned_comment_text'].apply(lambda x: filter_words(x, key_words))

# vectorize
tfidf_vectorizer = joblib.load('models/tfidf_vectorizer.joblib')
X_test_new = tfidf_vectorizer.transform(test['cleaned_comment_text'])
predictions = pd.DataFrame()
predictions['id'] = test['id']

# load model
models = {}
for label in columns_to_analyze:
    filename = f'models/{label}.joblib'
    models[label] = joblib.load(filename)

# predication
for label, model in models.items():
    predictions[label] = model.predict(X_test_new)

# save
predictions.to_csv('data/submission.csv', index=False)
