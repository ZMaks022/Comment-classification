import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


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


def delete_empty(data):
    if 'cleaned_comment_text' not in data.columns:
        raise KeyError

    data = data.dropna()
    return data[data['cleaned_comment_text'].str.strip().astype(bool)]
