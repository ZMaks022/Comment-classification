from src.prepare_text import filter_words
import pytest


@pytest.mark.parametrize("text, key_words, result", [("123 456 789", ["123", "789"], "123 789"),
                                                     ("asd gh", ["gh", "sdfs"], "gh"),
                                                     ("1 2 3 4 5 1", ["1"], "1 1"),
                                                     ("aaa bbb", [], ""),
                                                     ("aaaaa", ["bbbbb"], ""),
                                                     ("some text", ["some", "text"], "some text")
                                                     ])
def test_filter_good(text, key_words, result):
    assert filter_words("123 456 789", ["123", "789"]) == "123 789"


def test_large_text():
    text = "word " * 1000
    key_words = ["word"]
    assert filter_words(text, key_words) == text.strip()


def test_invalid_text_type():
    with pytest.raises(TypeError):
        filter_words(123, ["word"])


def test_invalid_key_words_type():
    with pytest.raises(TypeError):
        filter_words("some text", "word")


def test_invalid_key_word_type_in_list():
    with pytest.raises(TypeError):
        filter_words("some text", ["word", 123])
