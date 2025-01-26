from pyspark.ml.feature import Tokenizer, HashingTF, IDF, VectorAssembler
from pyspark.ml.regression import GBTRegressor, GBTRegressionModel
from pyspark.sql import DataFrame
from pyspark.sql.functions import concat_ws, when, col, size, split, udf, lower, regexp_replace
from pyspark.sql.types import FloatType
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from pyspark.sql.functions import abs as F_abs
import nltk
from nltk.data import find
import os
import datetime

def ensure_nltk_resources():
    try:
        find('vader_lexicon.zip')
    except LookupError:
        nltk.download('vader_lexicon')
    
    try:
        find('tokenizers/punkt.zip')
    except LookupError:
        nltk.download('punkt')
    
    try:
        find('corpora/stopwords.zip')
    except LookupError:
        nltk.download('stopwords')
    
    try:
        find('corpora/wordnet.zip')
    except LookupError:
        nltk.download('wordnet')

# Inizializza VADER
sia = SentimentIntensityAnalyzer()

def get_vader_sentiment(text):
    """
    Calcola il punteggio compound di VADER per il testo fornito.
    """
    if text:
        sentiment = sia.polarity_scores(text)
        return sentiment['compound']
    return 0.0

# Registra la funzione come UDF
vader_udf = udf(get_vader_sentiment, FloatType())

class CoherenceReviewModel:
    def __init__(self, df: DataFrame, model_path="models/gbt_model"):
        self.df = df
        self.model_path = model_path
        
        ensure_nltk_resources()
        
        # Carica il modello se esiste
        if os.path.exists(self.model_path):
            self.model = GBTRegressionModel.load(self.model_path)
            print(f"Modello caricato da: {self.model_path}")
        else:
            self.model = None
            print("Nessun modello trovato. Sarà necessario addestrarlo.")

    def preprocess_reviews(self):
        """
        Pulisce le recensioni positive e negative mantenendole separate.
        """
        # Pulizia delle recensioni positive
        self.df = self.df.withColumn("Positive_Review_Clean", lower(col("Positive_Review")))
        self.df = self.df.withColumn("Positive_Review_Clean", regexp_replace(col("Positive_Review_Clean"), r"[^a-z\s]", ""))

        # Pulizia delle recensioni negative
        self.df = self.df.withColumn("Negative_Review_Clean", lower(col("Negative_Review")))
        self.df = self.df.withColumn("Negative_Review_Clean", regexp_replace(col("Negative_Review_Clean"), r"[^a-z\s]", ""))

        # Calcolo dei punteggi di sentiment con VADER
        self.df = self.df.withColumn("Positive_Sentiment_Score", vader_udf(col("Positive_Review_Clean")))
        self.df = self.df.withColumn("Negative_Sentiment_Score", vader_udf(col("Negative_Review_Clean")))

        # Calcolo del punteggio netto di sentiment
        self.df = self.df.withColumn(
            "Net_Sentiment_Score",
            col("Positive_Sentiment_Score") - col("Negative_Sentiment_Score")
        )

        # Filtra recensioni brevi
        self.df = self.df.withColumn("Positive_Word_Count", size(split(col("Positive_Review_Clean"), r"\s+")))
        self.df = self.df.withColumn("Negative_Word_Count", size(split(col("Negative_Review_Clean"), r"\s+")))

        #self.df = self.df.filter((col("Positive_Word_Count") >= 5) | (col("Negative_Word_Count") >= 5))

    def train_sentiment_model(self):
        """
        Addestra un modello di GBT per predire il punteggio basato sul testo della recensione e salva il modello.
        """
        if self.model:
            print("Modello già addestrato e caricato. Salto l'addestramento.")
            return

        # Tokenizzazione
        tokenizer = Tokenizer(inputCol="Positive_Review_Clean", outputCol="words")
        words_data = tokenizer.transform(self.df)

        # TF-IDF
        hashing_tf = HashingTF(inputCol="words", outputCol="raw_features", numFeatures=2000)
        featurized_data = hashing_tf.transform(words_data)

        idf = IDF(inputCol="raw_features", outputCol="features")
        tfidf_model = idf.fit(featurized_data)
        tfidf_data = tfidf_model.transform(featurized_data)

        # Assemble le feature
        vector_assembler = VectorAssembler(
            inputCols=["features", "Positive_Word_Count", "Negative_Word_Count",
                       "Positive_Sentiment_Score", "Negative_Sentiment_Score", "Net_Sentiment_Score"],
            outputCol="features_vec"
        )

        prepared_data = vector_assembler.transform(tfidf_data)
        prepared_data = prepared_data.repartition(10)
        prepared_data.cache()

        # Gradient-Boosted Trees Regressor
        gbt_regressor = GBTRegressor(
            featuresCol="features_vec",
            labelCol="Reviewer_Score",
            maxDepth=10,
            maxBins=32,
            maxIter=50,
            stepSize=0.1
        )

        sampled_data = prepared_data.sample(fraction=0.5, seed=42)
        model = gbt_regressor.fit(sampled_data)

        # Salva il modello
        model.write().overwrite().save(self.model_path)
        print(f"Modello salvato in: {self.model_path}")

        self.model = model
        self.tfidf_data = prepared_data

    def analyze_consistency(self, threshold=2.0, n=10, export_path=None):
        """
        Analizza la coerenza tra le recensioni e i punteggi.
        """
        if self.model is None:
            raise ValueError("Il modello non è stato addestrato o caricato.")

        # Verifica se tfidf_data è disponibile; in caso contrario, preprocessa i dati
        if not hasattr(self, 'tfidf_data'):
            print("Preprocessing dei dati per analisi della coerenza...")
            tokenizer = Tokenizer(inputCol="Positive_Review_Clean", outputCol="words")
            words_data = tokenizer.transform(self.df)

            hashing_tf = HashingTF(inputCol="words", outputCol="raw_features", numFeatures=2000)
            featurized_data = hashing_tf.transform(words_data)

            idf = IDF(inputCol="raw_features", outputCol="features")
            tfidf_model = idf.fit(featurized_data)
            tfidf_data = tfidf_model.transform(featurized_data)

            vector_assembler = VectorAssembler(
                inputCols=["features", "Positive_Word_Count", "Negative_Word_Count",
                        "Positive_Sentiment_Score", "Negative_Sentiment_Score", "Net_Sentiment_Score"],
                outputCol="features_vec"
            )
            self.tfidf_data = vector_assembler.transform(tfidf_data)

        # Predici i punteggi
        predictions = self.model.transform(self.tfidf_data)

        # Vincola le predizioni all'intervallo 0-10
        predictions = predictions.withColumn(
            "prediction",
            when(col("prediction") < 0, 0).when(col("prediction") > 10, 10).otherwise(col("prediction"))
        )
        
        # Aggiungi penalizzazioni o incrementi basati su "No Positive" e "No Negative"    
        predictions = predictions.withColumn(
            "adjusted_prediction",
            when(col("Positive_Review_Clean") == "no positive", col("prediction") - 2.0)
            .when(col("Negative_Review_Clean") == "no negative", col("prediction") + 2.0)
            .otherwise(col("prediction"))
        )


        # Calcola l'errore assoluto
        predictions = predictions.withColumn("error", F_abs(col("prediction") - col("Reviewer_Score")))

        # Filtra le recensioni incoerenti
        inconsistent_reviews = predictions.filter(col("error") > threshold)
        
        # Converti la colonna Tags in stringa, se presente
        if "Tags" in inconsistent_reviews.columns:
            inconsistent_reviews = inconsistent_reviews.withColumn("Tags", concat_ws(", ", col("Tags")))
            
        if "words" in inconsistent_reviews.columns:
            inconsistent_reviews = inconsistent_reviews.withColumn("words", concat_ws(" ", col("words")))
            
        # Rimuovi colonne con tipi non supportati
        columns_to_drop = ["raw_features", "features", "features_vec"]
        for col_name in columns_to_drop:
            if col_name in inconsistent_reviews.columns:
                inconsistent_reviews = inconsistent_reviews.drop(col_name)

        #print(f"\nRecensioni incoerenti (errore > {threshold}):")
        #inconsistent_reviews.select("Positive_Review", "Negative_Review", "Reviewer_Score","prediction", "adjusted_prediction", "error").show(n, truncate=100)

        # Esporta le recensioni incoerenti, se richiesto
        if export_path:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            final_path = os.path.join(export_path, f"inconsistent_reviews_{timestamp}")
            inconsistent_reviews.write.mode("overwrite").csv(final_path, header=True)
            print(f"\nRecensioni incoerenti esportate in: {final_path}")