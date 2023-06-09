import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import nltk
from nltk.corpus import stopwords
import altair as alt

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load the data
df = pd.read_csv('D:/OneDrive/Рабочий стол/Fall 2022/DH/trump_tweets.csv', usecols=['id', 'text'])

# Preprocessing
def preprocess(text):
    # Lowercase
    text = text.lower()
    # Remove retweets
    if text.startswith('rt '):
        return ''
    # Remove links
    text = re.sub(r'https?:\/\/\S+', '', text)
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Remove stopwords
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

df['processed_text'] = df['text'].apply(preprocess)

# Remove rows with empty processed_text
df = df[df['processed_text'].str.strip() != '']

# Vectorize the text
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['processed_text'])

# Function to find most similar tweets
def find_similar_tweets(query):
    # Preprocess the query
    query = preprocess(query)
    # Vectorize the query
    query_vec = vectorizer.transform([query])
    # Compute cosine similarity between the query and all tweets
    similarity_scores = cosine_similarity(query_vec, X).flatten()
    # Get the top 10 most similar tweets
    top_tweet_indices = np.argsort(similarity_scores)[-10:]
    top_tweets = df.iloc[top_tweet_indices]
    top_tweets['similarity'] = similarity_scores[top_tweet_indices]
    return top_tweets[['text', 'similarity']]

# Ask the user for a query
#query = input("Enter a query: ")

#Or just manually enter a query here
query = ("I'm not a racist, but")

# Find the most similar tweets
similar_tweets = find_similar_tweets(query)

# Print each tweet on a new line
for i, row in similar_tweets.iterrows():
    print(f"Tweet: {row['text']}\nSimilarity: {row['similarity']}\n")

# Create an Altair chart
chart = alt.Chart(similar_tweets).mark_bar().encode(
    y=alt.Y('text:N', sort='-x', title='Tweet'),
    x=alt.X('similarity:Q', title='Similarity Score'),
    tooltip=['text', 'similarity']
).properties(
    title={
      "text": ["Similarity Scores of Top 10 Tweets"], 
      "subtitle": ["Based on cosine similarity with the query: " + query],
      "color": "black",
      "subtitleColor": "gray"
    }
)

chart.interactive()
