from pyspark.ml.feature import Tokenizer, HashingTF, IDF, VectorAssembler
from pyspark.ml.regression import GBTRegressor, GBTRegressionModel
from pyspark.sql import DataFrame
from pyspark.sql.functions import concat_ws, when, col, size, split, udf, lower, regexp_replace
from pyspark.sql.types import FloatType
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from pyspark.sql.functions import abs as F_abs
import nltk
import os
import datetime

nltk.download('vader_lexicon')
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

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
        """
        Inizializza la classe con il DataFrame preprocessato e un percorso per il modello.
        """
        self.df = df
        self.model_path = model_path
        
        # Carica il modello se esiste, altrimenti lascia il placeholder
        if os.path.exists(self.model_path):
            self.model = GBTRegressionModel.load(self.model_path)
            print(f"Modello caricato da: {self.model_path}")
        else:
            self.model = None
            print("Nessun modello trovato. Sarà necessario addestrarlo.")

    def preprocess_reviews(self):
        """
        Combina le recensioni positive e negative in una singola colonna di testo e pulisce i dati.
        """
        # Combina le recensioni
        self.df = self.df.withColumn(
            "Review_Text",
            concat_ws(" ",
                      when(col("Positive_Review") != "No Positive", col("Positive_Review")).otherwise(""),
                      when(col("Negative_Review") != "No Negative", col("Negative_Review")).otherwise("")
                      )
        )

        # Pulizia del testo
        self.df = self.df.withColumn("Review_Text", lower(col("Review_Text")))
        self.df = self.df.withColumn("Review_Text", regexp_replace(col("Review_Text"), r"[^a-z\s]", ""))

        # Calcola i punteggi di sentiment con VADER
        self.df = self.df.withColumn("Positive_Sentiment_Score", vader_udf(col("Positive_Review")))
        self.df = self.df.withColumn("Negative_Sentiment_Score", vader_udf(col("Negative_Review")))

        # Calcola il punteggio netto di sentiment
        self.df = self.df.withColumn(
            "Net_Sentiment_Score",
            col("Positive_Sentiment_Score") - col("Negative_Sentiment_Score")
        )

        # Rimuovi recensioni brevi
        self.df = self.df.withColumn("Word_Count", size(split(col("Review_Text"), r"\s+")))
        self.df = self.df.filter(col("Word_Count") >= 5)

    def train_sentiment_model(self):
        """
        Addestra un modello di GBT per predire il punteggio basato sul testo della recensione e salva il modello.
        """
        if self.model:
            print("Modello già addestrato e caricato. Salto l'addestramento.")
            return

        # Tokenizzazione
        tokenizer = Tokenizer(inputCol="Review_Text", outputCol="words")
        words_data = tokenizer.transform(self.df)

        # TF-IDF
        hashing_tf = HashingTF(inputCol="words", outputCol="raw_features", numFeatures=2000)
        featurized_data = hashing_tf.transform(words_data)

        idf = IDF(inputCol="raw_features", outputCol="features")
        tfidf_model = idf.fit(featurized_data)
        tfidf_data = tfidf_model.transform(featurized_data)

        # Aggiungi le colonne calcolate come feature
        tfidf_data = tfidf_data.withColumn(
            "Positive_Review_Word_Count",
            size(split(col("Positive_Review"), r"\s+"))
        ).withColumn(
            "Negative_Review_Word_Count",
            size(split(col("Negative_Review"), r"\s+"))
        )

        # Assemble le feature
        vector_assembler = VectorAssembler(
            inputCols=["features", "Positive_Review_Word_Count", "Negative_Review_Word_Count",
                       "Positive_Sentiment_Score", "Negative_Sentiment_Score", "Net_Sentiment_Score"],
            outputCol="features_vec"
        )

        prepared_data = vector_assembler.transform(tfidf_data)
        prepared_data = prepared_data.repartition(10)  # Migliora la distribuzione delle partizioni
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

        # Usa un campione bilanciato per l'addestramento
        sampled_data = prepared_data.sample(fraction=0.5, seed=42)

        # Addestra il modello
        model = gbt_regressor.fit(sampled_data)

        # Salva il modello
        model.write().overwrite().save(self.model_path)
        print(f"Modello salvato in: {self.model_path}")

        self.model = model
        self.tfidf_data = prepared_data

    def load_model(self):
        """
        Carica il modello salvato dal percorso specificato.
        """
        if os.path.exists(self.model_path):
            self.model = GBTRegressionModel.load(self.model_path)
            print(f"Modello caricato da: {self.model_path}")
        else:
            raise FileNotFoundError(f"Il modello non esiste nel percorso: {self.model_path}")

    def analyze_consistency(self, threshold=2.0, n=10, export_path=None):
        """
        Analizza la coerenza tra le recensioni e i punteggi.
        """
        if self.model is None:
            raise ValueError("Il modello non è stato addestrato o caricato.")

        # Predici i punteggi
        predictions = self.model.transform(self.tfidf_data)

        # Vincola le predizioni all'intervallo 0-10
        predictions = predictions.withColumn(
            "prediction",
            when(col("prediction") < 0, 0).when(col("prediction") > 10, 10).otherwise(col("prediction"))
        )

        # Calcola l'errore assoluto
        predictions = predictions.withColumn("error", F_abs(col("prediction") - col("Reviewer_Score")))

        # Filtra le recensioni incoerenti
        inconsistent_reviews = predictions.filter(col("error") > threshold)

        print(f"\nRecensioni incoerenti (errore > {threshold}):")
        inconsistent_reviews.select("Review_Text", "Reviewer_Score", "prediction", "error").show(n, truncate=100)

        # Distribuzione delle predizioni
        print("\nDistribuzione delle predizioni:")
        predictions.groupBy("prediction").count().orderBy("prediction").show(50, truncate=False)

        # Distribuzione degli errori
        print("\nDistribuzione degli errori:")
        predictions.groupBy("error").count().orderBy("error").show(50, truncate=False)

        # Esporta le recensioni incoerenti, se richiesto
        if export_path:
            # Aggiungi un timestamp al nome della directory
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            final_path = os.path.join(export_path, f"inconsistent_reviews_{timestamp}")

            inconsistent_reviews.select("Review_Text", "Reviewer_Score", "prediction", "error").write.mode("overwrite").csv(final_path, header=True)
            print(f"\nRecensioni incoerenti esportate in: {final_path}")

        # Restituisci un dizionario con i risultati principali
        results = {
            "predictions": predictions,
            "inconsistent_reviews": inconsistent_reviews,
            "error_distribution": predictions.groupBy("error").count().orderBy("error")
        }

        return results