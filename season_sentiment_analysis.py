from pyspark.sql.functions import col, month, when, udf, avg, concat
from pyspark.sql.types import FloatType
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from nltk.data import find

def ensure_nltk_resources():
    try:
        find('vader_lexicon.zip')
    except LookupError:
        nltk.download('vader_lexicon')
    
sia = SentimentIntensityAnalyzer()

def calculate_sentiment(text):
    if text:
        sentiment = sia.polarity_scores(text)
        return sentiment['compound']
    return 0.0

sentiment_udf = udf(calculate_sentiment, FloatType())

class SeasonSentimentAnalysis:
    def __init__(self, df):
        """
        Inizializza la classe con il DataFrame.
        """
        self.df = df
        ensure_nltk_resources()

    def preprocess(self):
        """
        Calcola il sentiment per recensioni positive e negative
        e aggiunge la stagione per ogni recensione.
        """
        # Calcola il sentiment per recensioni positive e negative
        '''self.df = self.df.withColumn("Positive_Sentiment", sentiment_udf(col("Positive_Review")))
        self.df = self.df.withColumn("Negative_Sentiment", sentiment_udf(col("Negative_Review")))'''
        self.df = self.df.withColumn("Total_Review", concat((col("Positive_Review"), col("Negative_Review"))))
        self.df = self.df.withColumn("Net_Sentiment", sentiment_udf(col("Total_Review")))

        # Calcola il sentiment netto
        #self.df = self.df.withColumn("Net_Sentiment", col("Positive_Sentiment") - col("Negative_Sentiment"))

        # Determina la stagione in base al mese
        self.df = self.df.withColumn(
            "Season",
            when((month(col("Review_Date")).isin(12, 1, 2)), "Winter")  # Inverno
            .when((month(col("Review_Date")).isin(3, 4, 5)), "Spring")  # Primavera
            .when((month(col("Review_Date")).isin(6, 7, 8)), "Summer")  # Estate
            .when((month(col("Review_Date")).isin(9, 10, 11)), "Autumn")  # Autunno
        )
        return self.df