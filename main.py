import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from random import shuffle

nltk.download([
     "names",
     "stopwords",
     "state_union",
     "twitter_samples",
     "movie_reviews",
     "averaged_perceptron_tagger",
     "vader_lexicon",
     "punkt",
 ])

tweets = [t.replace("://", "//") for t in nltk.corpus.twitter_samples.strings()]

sia = SentimentIntensityAnalyzer()

def is_positive(tweet: str) -> bool:
    """True if tweet has positive compound sentiment, False otherwise."""
    return sia.polarity_scores(tweet)["compound"] > 0

shuffle(tweets)
for tweet in tweets[:10]:
    print(">", is_positive(tweet), tweet)

print(sia.polarity_scores("i hate myself so much"))