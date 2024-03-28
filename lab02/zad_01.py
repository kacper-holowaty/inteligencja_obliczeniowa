# BŁĘDY I BRAKUJĄCE DANE W IRYSACH

import pandas as pd
df = pd.read_csv("iris_with_errors.csv")


def zlicz_puste():
    brakujące_dane = df.isnull().sum()
    print(brakujące_dane)

# zlicz_puste()

def sprawdz_zakres():
    column_names = ["sepal.length", "sepal.width", "petal.length", "petal.width"]
    for row in df.values:
        # print(row)
        for value in row:
            if value == '-':
                # if row[column_names[1]] ==
                # print(row[column_names[1, :]])


sprawdz_zakres()