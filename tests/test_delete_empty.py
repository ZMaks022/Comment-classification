from src.prepare_text import delete_empty
import pandas as pd
import pytest


def test_delete_empty_removes_empty_rows():
    data = pd.DataFrame({'cleaned_comment_text': ['test', '', ' ', 'another test']})
    result = delete_empty(data)
    assert len(result) == 2
    assert all(result['cleaned_comment_text'] == ['test', 'another test'])


def test_delete_empty_keeps_non_empty_rows():
    data = pd.DataFrame({'cleaned_comment_text': ['test', 'another test']})
    result = delete_empty(data)
    assert len(result) == 2
    assert all(result['cleaned_comment_text'] == ['test', 'another test'])


def test_delete_empty_with_all_empty_rows():
    data = pd.DataFrame({'cleaned_comment_text': ['', ' ', '   ']})
    result = delete_empty(data)
    assert result.empty


def test_delete_empty_with_empty_dataframe():
    data = pd.DataFrame({'cleaned_comment_text': pd.Series([], dtype="str")})
    result = delete_empty(data)
    assert result.empty


def test_delete_empty_without_required_column():
    data = pd.DataFrame({'other_column': ['value1', 'value2', 'value3']})
    with pytest.raises(KeyError):  # Или другое исключение, которое вы решите генерировать
        delete_empty(data)
