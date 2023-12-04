from src.prepare_text import prepare_text


def test_lowercase_conversion():
    assert prepare_text("TeSt") == "test"


def test_removal_of_non_alphabetic_characters():
    assert prepare_text("hello! 3world?") == "hello world"


def test_removal_of_short_words():
    assert prepare_text("a an the apple") == "apple"


def test_lemmatization():
    assert prepare_text("apples running") == "apple running"


def test_removal_of_stopwords():
    assert prepare_text("the apple is on the table") == "apple table"


def test_empty_string():
    assert prepare_text("") == ""


def test_combined_case():
    assert prepare_text("A 3Little cat! Running fast.") == "little cat running fast"
