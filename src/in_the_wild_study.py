import pandas as pd
import numpy as np
from train import model, vectorizer, emotion_labels
from preprocess import preprocess_text

# Load and combine datasets
df1 = pd.read_csv("data/news.csv", header=0)
df2 = pd.read_csv("data/letter.csv", header=0)
df = pd.concat([df1, df2], ignore_index=True)

# Preprocess the text
df['processed_text'] = df.iloc[:, 4].apply(preprocess_text)

# Feature extraction with TF-IDF
T = vectorizer.transform(df['processed_text'])

# Initialize counters and storage
category_counts = {category: 0 for category in emotion_labels}
top_tweet = {category: None for category in emotion_labels}
decision_scores_list = {category: [] for category in emotion_labels}  # For storing scores
total_tweets = len(df)

def Toint(value):
    """Convert string values to integers, handling 'k', 'm', and 'b' suffixes."""
    if pd.isna(value):
        return 0
    if isinstance(value, str):
        value = value.lower().strip()
        if value.endswith('k'):
            return int(float(value[:-1]) * 1000)
        elif value.endswith('m'):
            return int(float(value[:-1]) * 1000000)
        elif value.endswith('b'):
            return int(float(value[:-1]) * 1000000000)
        else:
            return int(value)
    else:
        return int(value)

def interaction_score(likes, retweets):
    return 3 * retweets + likes

# Loop through each tweet
for i, row in df.iterrows():
    tweet_vector = T[i]
    username = row[1]
    likes = Toint(row[5])
    retweets = Toint(row[6])

    # Get decision scores
    decision_scores = model.decision_function(tweet_vector)

    # Classify based on decision scores
    max_score = np.max(decision_scores)
    category = emotion_labels[np.argmax(decision_scores)]
    category_counts[category] += 1
    decision_scores_list[category].append(max_score)

    if max_score > 0.6:  # Filter tweets based on max_score
        score = interaction_score(likes, retweets)
        if top_tweet[category] is None or score > top_tweet[category][2]:
            top_tweet[category] = (row[4], username, score, max_score)

# Compute percentages and average scores
category_percentages = {category: (count / total_tweets) * 100 for category, count in category_counts.items()}
category_averages = {category: np.mean(decision_scores_list[category]) for category in emotion_labels}

# Output results
print(total_tweets)

for category in emotion_labels:
    print(f"Category: {category}")
    print(f"Percentage of tweets: {category_percentages[category]:.2f}%")
    print(f"Average max score: {category_averages[category]:.2f}")
    if top_tweet[category] is not None:
        tweet_text, username, score, max_score = top_tweet[category]
        print(f"Top tweet: {tweet_text} by {username} (Score: {score}) (Decision: {max_score})")
    else:
        print("No top tweet found")
    print()