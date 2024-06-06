# BAG OF WORDS

import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
from wordcloud import WordCloud

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

with open('wonderland.txt', 'r', encoding='utf-8') as file:
    text = file.read()

tokens = word_tokenize(text)

num_tokens = len(tokens)
print(f"Liczba słów po tokenizacji: {num_tokens}")

stop_words = set(stopwords.words('english'))

filtered_tokens = [word for word in tokens if word.lower() not in stop_words]

num_filtered_tokens = len(filtered_tokens)
print(f"Liczba słów po usunięciu stop-words: {num_filtered_tokens}")

additional_stop_words = []
for word in filtered_tokens:
    if word in string.punctuation or any(char.isdigit() for char in word):
        additional_stop_words.append(word)

stop_words.update(additional_stop_words)
stop_words.update(['“', '”', '’'])

filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
# filtered_tokens = set(filtered_tokens)
num_filtered_tokens = len(filtered_tokens)
print(f"Liczba słów po usunięciu kilku słów do stop-words i usunięciu tych stop-words: {num_filtered_tokens}")

lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
# lemmatized_tokens = set(lemmatized_tokens)

num_lemmatized_tokens = len(lemmatized_tokens)
print(f"Liczba słów po lematyzacji: {num_lemmatized_tokens}")

fd = FreqDist(lemmatized_tokens)

most_common_words = fd.most_common(10)
words, counts = zip(*most_common_words)

plt.figure(figsize=(10, 5))
plt.bar(words, counts)
plt.xlabel('Słowa')
plt.ylabel('Liczba wystąpień')
plt.title('10 najczęściej występujących słów')
plt.savefig('zad1_most_common_words.png')
plt.show()

wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(fd)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Chmura tagów')
plt.savefig('zad1_wordcloud.png') 
plt.show()