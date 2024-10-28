import pandas as pd
import spacy
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud

nlp = spacy.load("en_core_web_sm")

df1 = pd.read_csv("data/news.csv", header=0)
df2 = pd.read_csv("data/letter.csv", header=0)
df = pd.concat([df1, df2], ignore_index=True)

# Combine to string
tweets = df.iloc[:, 4].tolist()
text = ' '.join(tweets)
doc = nlp(text)

# Wordcloud noun
words = [token.text for token in doc if token.pos_ == 'NOUN' and not token.is_stop and len(token.text) > 2]
counts = Counter(words)
topWords = counts.most_common(30)

for word, count in topWords:
  print(f"{word}:{count}")

wordcloud = WordCloud(width=800, height=400, background_color='white', colormap = 'Blues').generate_from_frequencies(dict(topWords))
plt.figure(figsize=(8, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# Wordcloud verb
Vwords = [token.text for token in doc if token.pos_ == 'VERB' and not token.is_stop and len(token.text) > 2]
counts = Counter(Vwords)
topVwords = counts.most_common(30)

for word, count in topVwords:
  print(f"{word}:{count}")

wordcloud = WordCloud(width=800, height=400, background_color='white', colormap = 'PuRd_r').generate_from_frequencies(dict(topVwords))
plt.figure(figsize=(8, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()