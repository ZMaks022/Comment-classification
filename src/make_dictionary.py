from collections import Counter


def make_dictionary(data, columns_to_analyze):
    key_words = []

    for column in columns_to_analyze:
        toxic_comments = data[data[column] == 1]['cleaned_comment_text']

        all_toxic_comments = ' '.join(toxic_comments)
        toxic_word_frequencies = Counter(all_toxic_comments.split())
        most_common_toxic_words = toxic_word_frequencies.most_common(100)
        for word in most_common_toxic_words:
            if word[0] not in key_words:
                key_words.append(word[0])

    return key_words
