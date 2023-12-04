import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
import json
import joblib
import argparse
import sys
import requests
import os


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data', type=str, help='Path to a CSV file to process')
parser.add_argument('-o', '--output', type=str, help='Path to output of prediction')


if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)


args = parser.parse_args()


def prepare_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d', '', text)
    text = ' '.join([word for word in text.split() if len(word) > 2])
    lemmatizer = WordNetLemmatizer()
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])
    return text


def filter_words(text, key_words):
    if not isinstance(text, str):
        raise TypeError
    if not isinstance(key_words, list):
        raise TypeError
    for item in key_words:
        if not isinstance(item, str):
            raise TypeError

    return ' '.join([word for word in text.split() if word in key_words])


def download_file_from_github(url, local_filename):
    response = requests.get(url)

    if response.status_code == 200:
        with open(local_filename, 'wb') as f:
            f.write(response.content)
        print(f"File '{local_filename}' is successful  download.")
    else:
        print(f"Error downloading file: {response.status_code}")


args = parser.parse_args()

# create directory models
try:
    if not os.path.exists("models"):
        os.makedirs("models")
        print(f"Directory 'models' is successful created.")
    else:
        print(f"Directory 'models' already exist.")
except Exception as e:
    print(f"Error with creating directory: {e}")


# download pretrained models
columns_to_analyze = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
for label in columns_to_analyze:
    download_file_from_github(f"https://github.com/ZMaks022/ML_Lab4/blob/main/src/models/{label}.joblib?raw=true",
                              f"models/{label}.joblib")
models = {}
for label in columns_to_analyze:
    filename = f'models/{label}.joblib'
    models[label] = joblib.load(filename)

# download dictionary
download_file_from_github("https://github.com/ZMaks022/ML_Lab4/blob/main/src/data/key_words.json?raw=true",
                          "key_words.json")
key_words = []
with open('key_words.json', 'r') as file:
    key_words = json.load(file)

# download tfidf_vectorizer
download_file_from_github("https://github.com/ZMaks022/ML_Lab4/blob/main/src/models/tfidf_vectorizer.joblib?raw=true",
                          "tfidf_vectorizer.joblib")
tfidf_vectorizer = joblib.load('tfidf_vectorizer.joblib')

nltk.download('wordnet')
nltk.download('stopwords')

if args.data:
    test = pd.read_csv(args.data)
    test['cleaned_comment_text'] = test['comment_text'].apply(prepare_text)
    test['cleaned_comment_text'] = test['cleaned_comment_text'].apply(lambda x: filter_words(x, key_words))
    X_test_new = tfidf_vectorizer.transform(test['cleaned_comment_text'])
    predictions = pd.DataFrame()
    predictions['id'] = test['id']

    for label, model in models.items():
        predictions[label] = model.predict(X_test_new)
if args.output:
    predictions.to_csv(args.output, index=False)
