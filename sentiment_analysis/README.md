# Twitter Sentiment Analysis with Text Preprocessing and Machine Learning

This Python script performs sentiment analysis on Twitter data to classify tweets as racist/sexist (negative) or non-racist/sexist (positive). It also explores the use of natural language processing (NLP) techniques and machine learning models for sentiment classification.

**Features:**

- **Data Loading and Preprocessing:**
    - Loads a Twitter sentiment dataset from a CSV file.
    - Performs text cleaning, including removing patterns, special characters, numbers, punctuations, and short words.
    - Applies stemming to reduce words to their base form.
- **Exploratory Data Analysis (EDA):**
    - Visualizes the most frequent words using WordCloud for the entire dataset, positive tweets, and negative tweets.
    - Extracts and analyzes hashtags used in positive and negative tweets.
- **Machine Learning Model Training:**
    - Creates a bag-of-words (BOW) representation of the text data using CountVectorizer.
    - Splits the data into training and testing sets.
    - Trains Logistic Regression, Decision Tree, and K-Nearest Neighbors (KNN) models for sentiment classification.
    - Evaluates model performance using accuracy and F1-score metrics.
- **Basic Sentiment Analysis GUI (Optional):** (This section requires the `tkinter` library)
    - Provides a simple graphical user interface (GUI) to take user input and classify sentiment using a Naive Bayes model.

**Requirements:**

- `pandas` library: `pip install pandas`
- `numpy` library: `pip install numpy`
- `matplotlib` library: `pip install matplotlib`
- `seaborn` library: `pip install seaborn`
- `re` library (included in Python standard library)
- `string` library (included in Python standard library)
- `nltk` library: `pip install nltk`
- `wordcloud` library: `pip install wordcloud`
- `scikit-learn` library: `pip install scikit-learn`
- `tkinter` library (for the optional GUI, included in most Python installations)

**How to Run (Excluding GUI):**

1. Save the code as a Integrated Python Notebook file (e.g., `twitter_sentiment_analysis.ipynb`).
2. Open a terminal or command prompt and navigate to the directory where you saved the file.
3. Use notebooks like jupiter or google colab to run this file.
4. Execute the file and a popup is shown as tkinter module to take input and measure the tweet as pos or neg sentiment analysis.
