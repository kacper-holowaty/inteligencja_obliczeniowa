# TITANIC I REGUŁY ASOCJACYJNE

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import matplotlib.pyplot as plt

# Wczytanie danych oraz usunięcie pierwszej kolumny 
df = pd.read_csv('titanic.csv')
df = df.drop(df.columns[0], axis=1)

# Przekształcenie danych kategorycznych za pomocą One Hot Encoding
df_encoded = pd.get_dummies(df)

# Algorytm Apriori
frequent_itemsets = apriori(df_encoded, min_support=0.005, use_colnames=True)

# Wyszukanie reguł o sensownych parametrach
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.8)

# Posortowanie reguł wg ufności
rules = rules.sort_values(by='confidence', ascending=False)

# Wyświetlenie najciekawszych reguł
interesting_rules = rules[(rules['antecedents'].apply(lambda x: len(x)) == 2) & (rules['consequents'].apply(lambda x: len(x)) == 1)]
print(interesting_rules)

plt.figure(figsize=(10,5))
plt.scatter(interesting_rules['support'], interesting_rules['confidence'], alpha=0.5)
plt.xlabel('Support')
plt.ylabel('Confidence')
plt.title('Support vs Confidence')
plt.show()