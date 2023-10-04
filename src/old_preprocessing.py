import pandas as pd


def normalize(filename, location):
    print("\tpulling data...")
    data_rows = list()
    col_names = list()
    with open(filename, "r") as file:
        lines = file.readlines()
        for line in lines:
            if len(col_names) == 0:
                col_names = line.replace("\n", "").split(",")
            else:
                data_rows.append(line.replace("\n", "").split(",")[:-1])
    data = pd.DataFrame(data_rows, columns=col_names)

    for col in data.columns:
        try:
            data[col] = data[col].astype("float64")
        except ValueError:
            pass

    print("\tnormalizing...")
    data_info = data.describe()
    for col in data.columns:
        if data[col].dtype != "object":
            mean = data_info[col]["mean"]
            min_val = data_info[col]["min"]
            max_val = data_info[col]["max"]
            rng = max(abs(min_val), abs(max_val))

            data[col] = list(map(lambda x: 0 if rng == 0 else round((x - mean) / rng, 3), data[col].to_list()))

    print("\texpanding cols...")
    string_cols = list()
    fields = dict()
    for col in data.columns:
        if data[col].dtype == "object":
            if col == "class":
                continue
            string_cols.append(col)
            for i in range(len(data)):
                s = data[col][i]
                if s in fields.keys():
                    fields[s][i] = 1
                else:
                    fields[s] = zeroes(len(data))
                    fields[s][i] = 1

    for field in fields.keys():
        data[field] = fields[field]

    print("\tremoving cols...")
    for col in string_cols:
        if col == "class":
            continue
        data = data.drop(col, axis=1)

    data.to_csv(location)


def zeroes(length):
    lst = []
    for i in range(length):
        lst.append(0)
    return lst
