import pandas as pd
import numpy as np
import re
import nltk
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize the stemmer once globally
stemmer = PorterStemmer()

def remove_pattern(input_txt, pattern):
    """
    Removes a specific regex pattern (like twitter handles @user) from text.
    """
    r = re.findall(pattern, input_txt)
    for word in r:
        input_txt = re.sub(word, "", input_txt)
    return input_txt

def clean_and_stem_text(series):
    """
    Applies the full cleaning and stemming pipeline to a Pandas Series of tweets.

    The pipeline includes:
    1. Removing Twitter handles (@user)
    2. Removing special characters, numbers, and punctuation
    3. Removing short words (length <= 3)
    4. Tokenizing the text
    5. Stemming all tokens (PorterStemmer)
    6. Rejoining tokens into a single clean string.

    Args:
        series (pd.Series): A column containing the raw tweets.

    Returns:
        pd.Series: A column containing the fully cleaned and stemmed tweets.
    """
    # 1. Remove twitter handles (@user)
    print("Step 1/5: Removing Twitter handles...")
    cleaned_tweets = np.vectorize(remove_pattern)(series, "@[\w]*")

    # 2. Remove special characters, numbers, and punctuations
    print("Step 2/5: Removing special characters and numbers...")
    # Note: re.sub is used here for direct string replacement
    # '[^a-zA-Z#]' pattern from original code replaces everything but letters and '#' with a space
    cleaned_tweets = [re.sub(r'[^a-zA-Z#]', ' ', tweet) for tweet in cleaned_tweets]
    
    # 3. Remove short words (length <= 3) and split into tokens
    print("Step 3/5: Removing short words and tokenizing...")
    tokenized_tweet = pd.Series(cleaned_tweets).apply(
        lambda x: [w for w in x.split() if len(w) > 3]
    )

    # 4. Stem the words
    print("Step 4/5: Stemming tokens...")
    tokenized_tweet = tokenized_tweet.apply(
        lambda sentence: [stemmer.stem(word) for word in sentence]
    )

    # 5. Combine words into single sentence
    print("Step 5/5: Rejoining tokens into final text...")
    final_tweets = tokenized_tweet.apply(lambda x: " ".join(x))
    
    return final_tweets

def hashtag_extract(series):
    """
    Extracts all hashtags from a series of tweets.
    
    Args:
        series (pd.Series): A column containing the tweets.

    Returns:
        list: A flattened list containing all extracted hashtags.
    """
    hashtags = []
    for tweet in series:
        ht = re.findall(r"#(\w+)", tweet)
        hashtags.append(ht)
    return sum(hashtags, []) # Unnest list

def visualize_wordcloud(text_series, title="Word Cloud", figsize=(15, 8)):
    """
    Generates and plots a WordCloud for a series of text.
    
    Args:
        text_series (pd.Series): The text to visualize (e.g., df['clean_tweet']).
        title (str): The title for the plot.
        figsize (tuple): Size of the Matplotlib figure.
    """
    all_words = " ".join([sentence for sentence in text_series])
    wordcloud = WordCloud(width=800, height=500, random_state=42, max_font_size=100).generate(all_words)

    plt.figure(figsize=figsize)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.show()

def plot_top_hashtags(hashtag_list, title="Top 10 Hashtags", n=10, figsize=(15, 9)):
    """
    Calculates and plots the frequency distribution of the top N hashtags.

    Args:
        hashtag_list (list): The flattened list of hashtags.
        title (str): The title for the plot.
        n (int): The number of top hashtags to display.
        figsize (tuple): Size of the Matplotlib figure.
    """
    freq = nltk.FreqDist(hashtag_list)
    d = pd.DataFrame({'Hashtag': list(freq.keys()),
                      'Count': list(freq.values())})

    # Select top N hashtags
    d = d.nlargest(columns='Count', n=n)

    plt.figure(figsize=figsize)
    sns.barplot(data=d, x='Hashtag', y='Count')
    plt.title(title)
    plt.show()


if __name__ == '__main__':
    # Example usage (for testing the module directly)
    # This block requires a 'Twitter Sentiments.csv' file in the same directory.
    try:
        df = pd.read_csv('Twitter Sentiments.csv')
        # Download necessary NLTK data if not present (only needs to be run once)
        # nltk.download('punkt')
        
        df['clean_tweet'] = clean_and_stem_text(df['tweet'])

        print("\n--- Preprocessing complete. Running visualization example. ---")
        
        # Visualize positive hashtags
        ht_positive = hashtag_extract(df['clean_tweet'][df['label']==0])
        plot_top_hashtags(ht_positive, title="Top 10 Positive Hashtags")

    except FileNotFoundError:
        print("\nNote: Cannot run internal test. Please ensure 'Twitter Sentiments.csv' is present.")