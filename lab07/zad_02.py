# WYDŹWIĘK RECENZJI

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import text2emotion as te
nltk.download('vader_lexicon')

negative = """
They gave me a room (including door code) for a room that was already taken by another guest! 
When I entered, she screamed in fear. It was an awful experience. We called support and they 
moved me to a room in a different house (10 minute walk). The new house was very messy, with 
empty wine bottles everywhere, and the other guests were loud until 3 am. I didn't like being 
told to go into a taken room, I didn't like that there were empty wine bottles everywhere, and 
I didn't like that the other guests were loud.
"""
positive = """
This lodge was surprisingly more than I expected. The staff was really nice and
welcoming, our room was with two queen beds and balcony, the room was absolutely clean
upon arrival, the complimentary coffee and tea bags were really good, and I would say a
nice touch from them. We enjoyed the heated indoor pool, the lounge they have is really
beautiful. Overall, the interior design of the The whole lodge is from a really exquisite taste.
We ended up staying 1 extra night, and we loved it! Big shout out to Aida, who was always so nice
and helpful.
"""

sid = SentimentIntensityAnalyzer()

positive_scores = sid.polarity_scores(positive)

print("Opinia pozytywna:")
print("Negatywny:", positive_scores['neg'])
print("Neutralny:", positive_scores['neu'])
print("Pozytywny:", positive_scores['pos'])
print("Wynik zagregowany (compound):", positive_scores['compound'])
print()


negative_scores = sid.polarity_scores(negative)

print("Opinia negatywna:")
print("Negatywny:", negative_scores['neg'])
print("Neutralny:", negative_scores['neu'])
print("Pozytywny:", negative_scores['pos'])
print("Wynik zagregowany (compound):", negative_scores['compound'])


emotions_positive = te.get_emotion(positive)
emotions_negative = te.get_emotion(negative)

print("Emocje w opinii pozytywnej:")
print(emotions_positive)
print()

print("Emocje w opinii negatywnej:")
print(emotions_negative)