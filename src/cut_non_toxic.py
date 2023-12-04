

def cut_non_toxic(data, num_rows, toxic_columns):
    for col in toxic_columns:
        if col not in data.columns:
            raise KeyError

    non_toxic_rows = data[data[toxic_columns].sum(axis=1) == 0]

    if num_rows > len(non_toxic_rows):
        raise ValueError

    rows_to_remove = non_toxic_rows.sample(n=num_rows)
    data = data.drop(rows_to_remove.index)

    return data
