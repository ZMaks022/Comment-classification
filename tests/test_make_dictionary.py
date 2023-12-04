from src.make_dictionary import make_dictionary
import pandas as pd


def test_key_words_extraction():
    data = pd.DataFrame({
        'toxic': [1, 0, 1],
        'cleaned_comment_text': ['word1 word2', 'word3 word4', 'word2 word5']
    })
    expected_words = ['word1', 'word2', 'word5']
    assert set(make_dictionary(data, ['toxic'])) == set(expected_words)


def test_empty_dataframe_with_column():
    data = pd.DataFrame({'toxic': [], 'cleaned_comment_text': []})
    assert make_dictionary(data, ['toxic']) == []


def test_unique_key_words():
    data = pd.DataFrame({
        'toxic': [1, 0],
        'severe_toxic': [0, 1],
        'cleaned_comment_text': ['word1 word2', 'word2 word3']
    })
    expected_words = ['word1', 'word2', 'word3']
    assert set(make_dictionary(data, ['toxic', 'severe_toxic'])) == set(expected_words)


def test_limit_of_key_words():
    words = ['word' + str(i) for i in range(200)]
    data = pd.DataFrame({
        'toxic': [1] * 200,
        'cleaned_comment_text': [' '.join(words)]*200
    })
    key_words = make_dictionary(data, ['toxic'])
    assert len(key_words) == 100
    assert all(word in words for word in key_words)
