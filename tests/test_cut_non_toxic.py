from src.cut_non_toxic import cut_non_toxic
import pandas as pd
import pytest


def test_remove_specified_number_of_rows():
    data = pd.DataFrame({
        'toxic': [0, 0, 1, 0, 1],
        'cleaned_comment_text': ['text1', 'text2', 'text3', 'text4', 'text5']
    })
    original_length = len(data)
    num_rows = 2
    result = cut_non_toxic(data, num_rows, ['toxic'])
    assert len(result) == original_length - num_rows


def test_remove_non_toxic_rows():
    data = pd.DataFrame({
        'toxic': [0, 0, 1, 0, 1],
        'cleaned_comment_text': ['text1', 'text2', 'text3', 'text4', 'text5']
    })
    num_rows = 2
    result = cut_non_toxic(data, num_rows, ['toxic'])
    assert result['toxic'].sum() > data['toxic'].sum() - num_rows


def test_more_rows_requested_than_available():
    data = pd.DataFrame({
        'toxic': [0, 1, 1],
        'cleaned_comment_text': ['text1', 'text2', 'text3']
    })
    num_rows = 2
    with pytest.raises(ValueError):
        cut_non_toxic(data, num_rows, ['toxic'])


def test_empty_dataframe():
    data = pd.DataFrame(columns=['toxic', 'cleaned_comment_text'])
    num_rows = 2
    with pytest.raises(ValueError):
        cut_non_toxic(data, num_rows, ['toxic'])


def test_missing_toxic_columns():
    data = pd.DataFrame({
        'toxic': [0, 1],
        'cleaned_comment_text': ['text1', 'text2']
    })
    num_rows = 1
    with pytest.raises(KeyError):
        cut_non_toxic(data, num_rows, ['missing_column'])


