# BŁĘDY I BRAKUJĄCE DANE W IRYSACH

import difflib
import pandas as pd
import numpy as np

df = pd.read_csv("iris_with_errors.csv")

def zlicz_puste():
    df.replace('-', np.nan, inplace=True)
    brakujace_dane = df.isnull().sum()
    print(brakujace_dane)


def popraw_dane_numeryczne():
    for column in df.columns[:-1]:
        df[column] = pd.to_numeric(df[column], errors='coerce')
        df[column] = df[column].fillna(df[column].median())
        df[column] = df[column].apply(lambda x, column=column: x if 0 < x < 15 else df[column].median())

def popraw_nieprawidlowe_gatunki():
    prawidlowe_gatunki = ["Setosa", "Versicolor", "Virginica"]
    unikalne_gatunki = df['variety'].unique()
    
    for gatunek in unikalne_gatunki:
        if gatunek not in prawidlowe_gatunki:
            najblizszy = difflib.get_close_matches(gatunek, prawidlowe_gatunki, n=1, cutoff=0.6)
            if najblizszy:
                print(f'Znaleziono nieprawidłowy gatunek: {gatunek}. Zastępuję go gatunkiem: {najblizszy[0]}.')
                df.loc[df['variety'] == gatunek, 'variety'] = najblizszy[0]
            else:
                print(f'Nie można znaleźć podobnego gatunku dla: {gatunek}.')

zlicz_puste()
popraw_dane_numeryczne()
popraw_nieprawidlowe_gatunki()
zlicz_puste()
