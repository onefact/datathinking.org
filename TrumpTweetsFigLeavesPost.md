# Trump, Tweets and Fig leaves 
![Title datathinking](https://github.com/onefact/datathinking.org/assets/125737942/dcf331db-7e79-4257-ba57-aae55d78e96e)

## Author Information
|     | Name        | Surname    | Email                        |
| --- | ----------- | ---------- | ---------------------------- |
| 1   | Nikolai     | Shurakov   | nikolai.shurakov@ut.ee       |
| 2   | ChatGPT     |            | [OpenAI ChatGPT](https://chat.openai.com/) |

## Introduction
In this blog post, we delve into the world of tweets, specifically focusing on those authored by Donald Trump, the 45th President of the United States. The aim is to explore the concept of "fig leaves" in the context of Trump's tweets. Fig leaves, a term borrowed from the philosophy of language, refer to utterances or actions that provide cover for statements or behaviors that would otherwise be seen as racist. They are linguistic devices used to navigate conversations where explicit racism would be unacceptable or damaging.

We will be using a dataset of [Trump's tweets](https://www.kaggle.com/datasets/headsortails/trump-twitter-archive) and applying data science techniques to create a search engine. This search engine will be capable of finding tweets relevant to a given topic, thereby enabling us to analyze and understand the use of fig leaves in Trump's discourse on Twitter.

## Importance and Motivation
The motivation behind this project is twofold. Firstly, I am intrigued by the challenge of applying machine learning and natural language processing techniques to real-world, complex datasets. Tweets, with their brevity and colloquial language, present a unique challenge for text analysis.

Secondly, and perhaps more importantly, this project is driven by a desire to shed light on the subtle ways in which language can be used to mask or deflect from problematic statements or actions. In an era where social media platforms like Twitter have become powerful tools for political communication, understanding these linguistic strategies is crucial.

The stakes involved in this project are high. By developing a tool that can identify and analyze the use of fig leaves, we can contribute to a more informed and critical public discourse. Furthermore, this tool can potentially be adapted to analyze other public figures' social media discourse (Joe Biden, Elon Mask, choose your favourite twitter-account), providing valuable insights into their communication strategies.

## Rationale
### What is Known
The use of language as a tool to shape public perception is a well-studied phenomenon in the fields of linguistics, communication studies, and political science. In the context of social media, and Twitter in particular, it is known that these platforms have become significant tools for political communication. Politicians, including Donald Trump, have used Twitter to bypass traditional media and directly reach their followers. The language used in these tweets can influence public opinion and shape political discourse.

### What is Unknown
While the concept of fig leaves is less explored. My project aims to find some instances of fig leaves in a chosen dataset. In this way, it contributes to a future investigation of this phenomenon. 

### Decisions to be Made
The primary decision to be made in this project is the selection of appropriate data science techniques to analyze the dataset of tweets. This includes decisions about data cleaning, text preprocessing, vectorization, and the specific machine learning algorithms to be used.

Another important decision is the choice of search terms or topics to be used in testing the search engine. These should be chosen to effectively demonstrate the ability of the search engine to identify relevant tweets and analyze the use of fig leaves.

### Decision Makers
The decisions in this project will primarily be made by me but I will consult [ChatGPT](https://chat.openai.com/). 

## What are fig leaves in philosophy of language?
The term "figleaves" in the context of this paper - [Racial Figleaves, the Shifting Boundaries of the Permissible, and the Rise of Donald Trump](https://www.jstor.org/stable/26529439) by *Jennifer M. Saul* refers to utterances or actions that provide cover for statements or behaviors that would otherwise be seen as racist. They are used to avoid a confrontation with the possibility that something racist is occurring. Here are some key excerpts from the paper that explain the concept:

1. "A racial figleaf is an utterance made in addition to one that would otherwise be seen as racist. Unlike in the case of an implicit appeal/covert dogwhistle, race has been explicitly mentioned. The figleaf provides cover for what would otherwise have too much potential to be labeled as racist." (Page 7)

2. "A synchronic figleaf is one provided at roughly the same time as the utterance for which it is a figleaf. Probably the most easily recognizable figleaf is the classic “I’m not a racist but . . . ,” followed by something explicitly racial and quite possibly explicitly racist." (Page 7)
 
3. "The figleaf offers some way of avoiding a confrontation with the possibility that something racist is going on. How well this works varies a great deal from context to context and audience to audience." (Page 7)
4. "A racial figleaf is, generally speaking, an attempt to block an inference from the fact that the speaker has made an openly racist utterance R to a claim like (11): (11) The speaker is racist." (Page 11)
5. "We must, I think, point to the figleaves as figleaves, and explain their power to distort. It is vitally important that we maintain a firm focus on what is being said and done, rather than letting the conversation drift to what some additional utterance might indicate about racism “in the heart.” (Page 18)

Examples of fig leaves:
- "I'm not a racist, but...",
- "Some of my best friends are [insert race/ethnicity]...",
- "I don't see color...",
- "I don't have a racist bone in my body...",
- "I can't be racist, I voted for Obama...",
- "I don’t care if you're white, black, yellow, green or purple..."

In summary, "figleaves" are used to mask or deflect from racist statements or actions, allowing the speaker to maintain a semblance of non-racism. They can be used strategically to navigate conversations and situations where explicit racism would be unacceptable or damaging.

## Metadata
The metadata comes from the author of the dataset - [Heads or Tails](https://www.kaggle.com/headsortails) user of Kaggle: 

>**Context**
The former US president Donald Trump was notoriously active on Twitter. On January 8th, 2021, the platform decided to suspend his account, citing ["the risk of further incitement of violence"](https://blog.twitter.com/en_us/topics/company/2020/suspension.html) following the violent riots at the US Capitol building on Jan 6th. Trump's Twitter activity constitutes an important documentation of escalating polarisation in the US political and societal discourse during the second decade of the 2000s.

>**Content**
This dataset contains all of Trump's tweets since 2009. It was copied in its entirety from the website [The Trump Archive](https://www.thetrumparchive.com/) who did all the work in periodically scraping Trump's Twitter account until his suspension in 2021. All I added was some light cleaning of column names and some equally light text formatting adjustments. There are several other Trump Tweet datasets on Kaggle, but I didn't see one that was as complete or recent as this archive. On the completeness of the archive, the [website FAQ](https://www.thetrumparchive.com/faq) notes that "the site launched in September 2016. If [Trump] deleted a tweet before that, it won't be in here. If he deleted a tweet since then, it should be in here."

>**Acknowledgements**
All the credit goes to Brendan from [The Trump Archive](https://www.thetrumparchive.com/) who compiled this data and made it publicly available.

## Script
```python
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
df = pd.read_csv('trump_tweets.csv', usecols=['id', 'text'])

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
```
## Maths
Here are the key mathematical concepts used in this script:

1. **TF-IDF Vectorization**:

The TF-IDF value for a word in a document is calculated by the following formula:

$$ \text{TFIDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D) $$

where:
- $\(\text{TF}(t, d)\)$ is the Term Frequency, i.e., the number of times term \(t\) appears in document \(d\)
- $\(\text{IDF}(t, D)\)$ is the Inverse Document Frequency, calculated as:

$$ \text{IDF}(t, D) = \log \left( \frac{N}{\text{DF}(t, D)} \right) $$

where:
- $\(N\)$ is the total number of documents in the corpus
- $\(\text{DF}(t, D)\)$ is the number of documents in the corpus that contain term \(t\)

2. **Cosine Similarity**:

The cosine similarity between two vectors \(A\) and \(B\) is calculated as follows:

$$ \text{cosine similarity}(A, B) = \frac{(A \cdot B)}{\lVert A \rVert \cdot \lVert B \rVert} $$

where:
- $\(A \cdot B)\$ is the dot product of $\(A\)$ and $\(B\)$
- $\||A\||\$ is the 2-norm (Euclidean length) of vector $\(A)\$
- $\||B\||\$ is the 2-norm (Euclidean length) of vector $\(B)\$

3. **K-Means Clustering**:

The objective function of K-means is:

$$ J = \sum_{i=1}^{k} \sum_{x \in C_i} \|x - \mu_i\|^2 $$

where:
- $\(C_i\)$ is the set of points that belong to cluster $\(i\)$
- $\(\mu_i\)$ is the centroid of cluster $\(i\)$
- $\(\|x - \mu_i\|^2\)$ is the squared Euclidean distance between a point $\(x\)$ and the centroid $\(\mu_i\)$

## Visualization
All visualizations are interactive Altair visualizations available in an open repository [here](https://github.com/nshut/datathinking_blogpost_TrumpTweetsFigLeaves/tree/2808897fae7cf8e0ebb65f01fcca228ceff6663b)
I include png images to show how they look like:
![1Iamnotaracist](https://github.com/onefact/datathinking.org/assets/125737942/f3b4edd9-2729-4509-8f90-f07cb6a1f31a)
![2myfrinedsare black](https://github.com/onefact/datathinking.org/assets/125737942/668b9d77-9f18-44ba-924c-3981a4140d6d)
![3myfrinedsarespanish](https://github.com/onefact/datathinking.org/assets/125737942/26472778-730b-4bfc-a421-a7944f9ac0be)
![4idontseecolour](https://github.com/onefact/datathinking.org/assets/125737942/49cc0d06-24e4-44cd-8976-7f3021df4b64)
![5racistbone](https://github.com/onefact/datathinking.org/assets/125737942/1f2e561c-988b-488b-88e3-34483bb3e882)
![6cantberacist](https://github.com/onefact/datathinking.org/assets/125737942/8b029a5e-d400-4eb2-95ad-5c7d92731ba7)
![7dontcareblackyellow](https://github.com/onefact/datathinking.org/assets/125737942/3e480aed-0ecd-46cf-b338-b3ae1afb32a5)
 
## Some results and thoughts on the research question
we can see several instances where the concept of "figleaves" as defined in Jennifer M. Saul's paper might apply. Here are a few examples:

- Tweet: _"Those Tweets were NOT Racist. I don’t have a Racist bone in my body! The so-called vote to be taken is a Democrat con game. Republicans should not show “weakness” and fall into their trap. This should be a vote on the filthy language, statements and lies told by the Democrat....."_
Here, the phrase "I don’t have a Racist bone in my body!" could be seen as a figleaf. It's a statement that's meant to deflect from the possibility that the preceding or following statements could be interpreted as racist.

- Tweet: _"Unemployment for Black Americans is the lowest ever recorded. Trump approval ratings with Black Americans has doubled. Thank you, and it will get even (much) better! @FoxNews"_
The claim about low unemployment for Black Americans and doubled approval ratings could be seen as a figleaf. These statements could be interpreted as an attempt to deflect from potential criticisms of racism by pointing to positive outcomes for Black Americans during his presidency.

- Tweet: _"For 47 years, Joe Biden viciously attacked Black Americans. He called young black men “super predators.” To every black American: I am asking for your vote. This is your one and only chance to show Sleepy Joe what you think of his decision to attack you, jail you, and betray you!"_
The accusation against Joe Biden could be seen as a figleaf. By accusing Biden of attacking Black Americans, the tweet could be attempting to deflect from any potential criticisms of racism against Trump himself.

These examples demonstrate how the concept of "figleaves" can be applied to real-world texts like tweets. It's important to note that the identification of figleaves can be somewhat subjective and depends on the context of the statement and the interpretation of the reader.


## Reflection on emotional component

Embarking on this project has been an emotional journey, filled with moments of frustration, confusion, and ultimately, accomplishment. As a graduate student in philosophy, I found myself in unfamiliar territory. Programming and data science were new fields to me, and the learning curve was steep.

There were times when I felt lost, unsure of how to proceed. The complexity of the task at hand often seemed overwhelming. But in those moments of doubt, I found support in unexpected places. ChatGPT, an AI developed by OpenAI, became an invaluable resource, providing guidance and assistance when I needed it most.

I also owe a debt of gratitude to my friend, Mykyta Luzan. His help was instrumental in navigating the challenges of this project. His patience and expertise helped me overcome many obstacles along the way.

Despite the challenges, I persevered. I poured my effort and thought into this project, driven by the belief that it could shed light on important aspects of public discourse. The process was difficult, but it was also rewarding. I learned not just about programming and data science, but also about resilience and the power of persistence.

In the end, I hope that this project will be of value to others. It demonstrates the potential of data science techniques to analyze and understand complex linguistic phenomena. More importantly, it serves as a testament to the fact that with determination, support, and a willingness to learn, it's possible to venture into new territories and achieve meaningful results.

This journey has taught me that the path to knowledge is often fraught with challenges. But with perseverance, and with the help of friends and AI companions like ChatGPT, these challenges can be overcome. 

## Conclusion
The search engine developed in this project has proven to be a valuable tool for identifying instances of "figleaves" in the tweets of Donald Trump. By leveraging natural language processing techniques, the engine was able to sift through large volumes of tweets and pinpoint instances where language was potentially being used to mask or deflect from racist statements or actions.

The implications of this project extend beyond the analysis of Trump's tweets. The search engine could be adapted to analyze the tweets of other public figures, such as Joe Biden or Elon Musk. This would provide valuable insights into their communication strategies and the potential use of figleaves in their discourse. By applying this tool across a range of public figures, we can gain a broader understanding of how language is used to navigate sensitive topics in public discourse.

This project underscores the importance of critical engagement with the language used by public figures. It highlights the subtle ways in which language can be used to mask or deflect from problematic statements or actions. By developing tools to identify and analyze these linguistic strategies, we can contribute to a more informed and critical public discourse.

However, there is always room for improvement. The current approach primarily captures lexical (word-level) similarities and does not account for more complex semantic or syntactic similarities that might exist between tweets. More advanced natural language processing techniques, such as word embeddings or transformer models, could be used to capture these more complex relationships. These methods can provide a more nuanced analysis but are also more computationally intensive and require more expertise to use effectively.

Additionally, the list of figleaf phrases used in this project is relatively basic and may not capture all potential figleaves. A more detailed list, perhaps developed with the input of experts in linguistics or communication studies, could improve the accuracy of the search engine.

## Project repository link
[https://github.com/nshut/datathinking_blogpost_TrumpTweetsFigLeaves/tree/f2b6eb8fcb8ca66a8bb5797b4da9cbfde68b84e9](https://github.com/nshut/datathinking_blogpost_TrumpTweetsFigLeaves)
