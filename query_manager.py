from spark_builder import SparkBuilder
from pyspark.sql import functions as F
from pyspark.sql.functions import explode, split, avg, col, desc,count
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from coherence_review_model_vader import CoherenceReviewModel
from season_sentiment_analysis import SeasonSentimentAnalysis


class QueryManager:
    def __init__(self, spark_builder: SparkBuilder):
        self.df = spark_builder.df_finale

    # ------------------------------QUERY 1----------------------------------------------

    def words_score_analysis(self, n=20):
        """
        Analizza quali parole nelle recensioni positive o negative sono indicatrici di punteggi alti o bassi.
        """
        df = self.df

        # Tokenizza le parole nelle recensioni positive
        positive_words = df.select(
            col("Reviewer_Score"),
            explode(split(col("Positive_Review"), r"\s+")).alias("word")
        ).filter(col("word") != "")  # Rimuove eventuali parole vuote

        # Calcola la media dello score per ogni parola positiva
        positive_word_scores = positive_words.groupBy("word") \
            .agg(avg("Reviewer_Score").alias("avg_score")) \
            .orderBy(desc("avg_score"))

        print(f"\nTop {n} parole positive con il punteggio più alto:")
        positive_word_scores.show(n, truncate=False)

        # Tokenizza le parole nelle recensioni negative
        negative_words = df.select(
            col("Reviewer_Score"),
            explode(split(col("Negative_Review"), r"\s+")).alias("word")
        ).filter(col("word") != "")  # Rimuove eventuali parole vuote

        # Calcola la media dello score per ogni parola negativa
        negative_word_scores = negative_words.groupBy("word") \
            .agg(avg("Reviewer_Score").alias("avg_score")) \
            .orderBy("avg_score")  # Ordina dal punteggio più basso

        print(f"\nTop {n} parole negative con il punteggio più basso:")
        negative_word_scores.show(n, truncate=False)
        
        
    # ------------------------------QUERY 2----------------------------------------------
    #Correlazione tra nazionalità e recensioni positive e negative.
    def nationality_review_analysis(self, n=20):
        """
        Analizza le nazionalità in base ai punteggi medi e alle recensioni positive/negative.
        """
        df = self.df

        # Nazionalità più "buone" (punteggi medi più alti)
        good_nationalities = df.groupBy("Reviewer_Nationality") \
            .agg(
                F.avg("Reviewer_Score").alias("avg_score"),
                F.count("*").alias("review_count")
            ).orderBy(F.desc("avg_score"))

        print(f"\nTop {n} nazionalità con il punteggio medio più alto:")
        good_nationalities.show(n, truncate=False)

        # Nazionalità più "cattive" (punteggi medi più bassi)
        bad_nationalities = df.groupBy("Reviewer_Nationality") \
            .agg(
                F.avg("Reviewer_Score").alias("avg_score"),
                F.count("*").alias("review_count")
            ).orderBy("avg_score")

        print(f"\nTop {n} nazionalità con il punteggio medio più basso:")
        bad_nationalities.show(n, truncate=False)

        # Correlazione tra nazionalità e lunghezza delle recensioni positive/negative (eliminabile, magari serve quindi lascio)
        nationality_reviews = df.groupBy("Reviewer_Nationality") \
            .agg(
                F.avg("Review_Total_Positive_Word_Counts").alias("avg_positive_words"),
                F.avg("Review_Total_Negative_Word_Counts").alias("avg_negative_words")
            ).orderBy(F.desc("avg_positive_words"))

        print(f"\nCorrelazione tra nazionalità e lunghezza delle recensioni positive/negative:")
        nationality_reviews.show(n, truncate=False)
        
    # ------------------------------QUERY 3----------------------------------------------

    #Influenza delle tag sullo scoring
    def tag_influence_analysis(self, n=10):
        """
        Analizza l'influenza delle tag sul punteggio dato.
        """
        df = self.df

        # Esplodi la colonna Tags in righe individuali
        exploded_tags = df.select(
            col("Reviewer_Score"),
            explode(col("Tags")).alias("tag")
        )

        # Calcola la media del punteggio per ciascuna tag
        tag_scores = exploded_tags.groupBy("tag") \
            .agg(
                F.avg("Reviewer_Score").alias("avg_score"),
                F.count("*").alias("tag_count")
            ).orderBy(F.desc("avg_score"))

        print(f"\nTop {n} tag con il punteggio medio più alto:")
        tag_scores.show(n, truncate=False)

        # Tag con punteggio più basso
        print(f"\nTop {n} tag con il punteggio medio più basso:")
        tag_scores.orderBy("avg_score").show(n, truncate=False)

        # Filtra per specifiche tag di interesse
        specific_tags = ["Solo traveler", "Business trip", "Leisure trip", "Couple"]
        filtered_tags = tag_scores.filter(col("tag").isin(specific_tags))

        print("\nInfluenza delle tag selezionate:")
        filtered_tags.show(truncate=False)
        
        
    # ------------------------------QUERY 4----------------------------------------------

    #correlazione tra il numero di recensioni passate di un recensore e il punteggio medio che tende a dare
    def reviewer_behavior_analysis(self):
        """
        Analizza la correlazione tra il numero di recensioni passate di un recensore 
        e il punteggio medio che tende a dare.
        """
        df = self.df

        # Calcola la media del punteggio dato rispetto al numero di recensioni fatte
        reviewer_analysis = df.groupBy("Total_Number_of_Reviews_Reviewer_Has_Given") \
            .agg(
                F.avg("Reviewer_Score").alias("avg_score"),
                F.count("*").alias("review_count")
            ).orderBy(F.desc("Total_Number_of_Reviews_Reviewer_Has_Given"))

        print("\nAnalisi del comportamento dei recensori (esperienza vs punteggio medio):")
        reviewer_analysis.show(50, truncate=False)
        
    # ------------------------------QUERY 5----------------------------------------------
    # Analisi degli outlier: identificare discrepanze significative tra il punteggio medio di un hotel e il numero di recensioni positive e negative.
    def outlier_analysis(self, score_threshold_high=8.5, score_threshold_low=5.0, review_threshold=10):
        """
        Identifica hotel con recensioni anomale:
        - Punteggio alto, ma molte recensioni negative.
        - Punteggio basso, ma molte recensioni positive.
        """
        df = self.df

        # Raggruppa per hotel e calcola metriche
        hotel_reviews = df.groupBy("Hotel_Address", "Hotel_Name") \
            .agg(
                F.avg("Reviewer_Score").alias("avg_score"),
                F.sum(F.when(F.col("Negative_Review") != "No Negative", 1).otherwise(0)).alias("total_negative_reviews"),
                F.sum(F.when(F.col("Positive_Review") != "No Positive", 1).otherwise(0)).alias("total_positive_reviews"),
                F.count("*").alias("review_count")
            )

        # Outlier 1: Hotel con punteggio alto e molte recensioni negative
        high_score_negative_reviews = hotel_reviews.filter(
            (col("avg_score") > score_threshold_high) &
            (col("total_negative_reviews") > review_threshold)
        )

        print("\nHotel con punteggio alto e molte recensioni negative:")
        high_score_negative_reviews.select(
            "Hotel_Name", "Hotel_Address", "avg_score", 
            "total_negative_reviews", "total_positive_reviews", "review_count"
        ).show(truncate=False)

        # Outlier 2: Hotel con punteggio basso e molte recensioni positive
        low_score_positive_reviews = hotel_reviews.filter(
            (col("avg_score") < score_threshold_low) &
            (col("total_positive_reviews") > review_threshold)
        )

        print("\nHotel con punteggio basso e molte recensioni positive:")
        low_score_positive_reviews.select(
            "Hotel_Name", "Hotel_Address", "avg_score", 
            "total_positive_reviews", "total_negative_reviews", "review_count"
        ).show(truncate=False)

    # ------------------------------QUERY 6----------------------------------------------
    #Analisi della lunghezza: verificare se recensioni più lunghe tendono ad essere più positive o negative.
    def review_length_analysis(self, n=20):
        """
        Analizza la lunghezza delle recensioni e verifica se recensioni più lunghe 
        tendono ad essere più positive o negative.
        """
        df = self.df
        df.filter((F.col("Review_Total_Positive_Word_Counts") > 0) | (F.col("Review_Total_Negative_Word_Counts") > 0)) #aggiunto per ottimizzare, filtro le recensioni vuote
        
        # Calcolo della lunghezza media delle recensioni positive e negative per punteggio
        review_length = df.groupBy("Reviewer_Score") \
            .agg(
                F.avg("Review_Total_Positive_Word_Counts").alias("avg_positive_length"),
                F.avg("Review_Total_Negative_Word_Counts").alias("avg_negative_length"),
                F.count("*").alias("review_count")
            ).orderBy("Reviewer_Score")

        print("\nLunghezza media delle recensioni positive e negative per punteggio:")
        review_length.show(n, truncate=False)

        # Analisi generale: recensioni più lunghe tendono ad essere più positive o negative?
        avg_positive = df.agg(F.avg("Review_Total_Positive_Word_Counts").alias("overall_avg_positive_length")).collect()[0][0]
        avg_negative = df.agg(F.avg("Review_Total_Negative_Word_Counts").alias("overall_avg_negative_length")).collect()[0][0]

        print(f"\nLunghezza media generale delle recensioni positive: {avg_positive:.2f}")
        print(f"Lunghezza media generale delle recensioni negative: {avg_negative:.2f}")
        
#--------------------------QUERY 7------------------------------------------------
#Analisi della coerenza tra recensioni e punteggi:predire il punteggio basato sul contenuto della recensione e confrontarlo con il punteggio effettivamente dato
    def coherence_analysis(self, threshold=2.0,n=10, export_path=None):
        """
        Richiama l'analisi del modello di sentiment e coerenza.
        """
        sentiment_model = CoherenceReviewModel(self.df)

        # Preprocessa il testo
        sentiment_model.preprocess_reviews()

        # Addestra il modello
        sentiment_model.train_sentiment_model()

        # Esegui l'analisi di coerenza
        predictions = sentiment_model.analyze_consistency(threshold,n, export_path) # treshold serve a defnire un limite massimo accettabile per l'errore assoluto tra il punteggio predetto dal modello e il punteggio reale
        
        
        
#--------------------------QUERY 8------------------------------------------------
#Durata della percezione negativa: analizzare se la percezione negativa diminuisce nel tempo.def recovery_time_analysis(self, n=20):
    def recovery_time_analysis(self, n=20):
        """
        Analizza quanto tempo ci vuole affinché un hotel passi da una recensione negativa (<6)
        a una recensione positiva (>6). Calcola medie per ogni fase e verifica il trend futuro.
        """
        df = self.df

        # Ordina le recensioni per hotel e data
        #window_spec = Window.partitionBy("Hotel_Name").orderBy("Review_Date")

        # Finestra di 10 recensioni precedenti per calcolare la media
        window_prev = Window.partitionBy("Hotel_Name").orderBy("Review_Date").rowsBetween(-10, 0)

        # Finestra di 5 recensioni successive per il trend dopo la positiva
        window_after_positive = Window.partitionBy("Hotel_Name").orderBy("Review_Date").rowsBetween(1, 5)

        # Calcolo di avg_after_negative (considerando tutte le recensioni precedenti)
        df = df.withColumn("avg_after_negative", F.avg("Reviewer_Score").over(window_prev))

        # Filtra recensioni negative
        df_negative = df.filter(F.col("Reviewer_Score") < 6).withColumnRenamed(
            "Reviewer_Score", "Negative_Score"
        ).withColumnRenamed("Review_Date", "Negative_Review_Date").withColumnRenamed(
            "avg_after_negative", "avg_after_negative_score"
        )

        # Filtra recensioni positive
        df_positive = df.filter(F.col("Reviewer_Score") > 6).withColumn(
            "avg_trend_after_positive",
            F.avg("Reviewer_Score").over(window_after_positive)
        ).withColumnRenamed("Reviewer_Score", "Positive_Score").withColumnRenamed(
            "Review_Date", "Positive_Review_Date"
        )

        # Join tra recensioni negative e positive
        df_recovery = df_positive.join(
            df_negative,
            on="Hotel_Name",
            how="inner"
        ).withColumn(
            "days_between",
            F.datediff(F.col("Positive_Review_Date"), F.col("Negative_Review_Date"))
        ).filter(F.col("days_between") > 0)  # Escludi casi senza recupero

        # Calcolo di avg_after_positive
        df_recovery = df_recovery.withColumn(
            "avg_after_positive",
            (F.col("avg_after_negative_score") + F.col("Positive_Score")) / 2
        )

        # Filtro per il days_between minimo per ogni hotel
        window_min_days = Window.partitionBy("Hotel_Name").orderBy("days_between")
        df_recovery_filtered = df_recovery.withColumn(
            "rank", F.row_number().over(window_min_days)
        ).filter(F.col("rank") == 1)  # Mantieni solo la riga con il days_between minimo

        # Calcola il tempo medio di recupero
        avg_recovery_time = df_recovery_filtered.agg(F.avg("days_between").alias("avg_recovery_time")).collect()[0][0]

        # Seleziona i risultati finali
        result = df_recovery_filtered.select(
            "Hotel_Name",
            "avg_after_negative_score",
            "Negative_Score",
            "days_between",
            "Positive_Score",
            "avg_after_positive",
            "avg_trend_after_positive"
        ).distinct().orderBy("Hotel_Name", "days_between")

        print("\nRisultati per ogni hotel:")
        result.show(n, truncate=False)
        print(f"\nTempo medio di recupero (giorni): {avg_recovery_time:.2f}")
        
        
#--------------------------QUERY 9------------------------------------------------

    #Analisi della reputazione: differenza tra il punteggio medio storico di un hotel e il punteggio medio delle recensioni recenti
    def reputation_analysis(self, recent_reviews=10, n=20):
        """
        Analizza la differenza tra il punteggio medio storico di un hotel e il punteggio medio delle recensioni recenti.
        """
        df = self.df

        # Calcolo del punteggio medio storico per ogni hotel
        avg_historical_window = Window.partitionBy("Hotel_Name")
        df = df.withColumn(
            "avg_historical_score",
            F.avg("Reviewer_Score").over(avg_historical_window)
        )

        # Calcolo del punteggio medio delle recensioni recenti (finestra di N recensioni)
        recent_reviews_window = Window.partitionBy("Hotel_Name").orderBy(F.desc("Review_Date")).rowsBetween(Window.currentRow, Window.currentRow + recent_reviews - 1)

        df = df.withColumn(
            "avg_recent_score",
            F.avg("Reviewer_Score").over(recent_reviews_window)
        )

        # Filtro per ottenere una riga per ogni hotel
        df_aggregated = df.groupBy("Hotel_Name").agg(
            F.first("avg_historical_score").alias("avg_historical_score"),
            F.first("avg_recent_score").alias("avg_recent_score")
        )

        # Calcolo della differenza tra il punteggio recente e quello storico
        df_aggregated = df_aggregated.withColumn(
            "score_difference",
            F.col("avg_recent_score") - F.col("avg_historical_score")
        )

        # Risultati finali
        print("\nRisultati dell'analisi della reputazione:")
        df_aggregated.orderBy("score_difference", ascending=False).show(n, truncate=False)

        # Media delle differenze
        avg_difference = df_aggregated.agg(F.avg("score_difference").alias("avg_difference")).collect()[0][0]
        print(f"\nDifferenza media complessiva: {avg_difference:.2f}")
                
#-----------------------QUERY 10------------------------------------------------
    def seasonal_sentiment_analysis(self, n=4):
        """
        Analizza come il sentiment medio delle recensioni varia in base alla stagione dell'anno.
        """
        # Preprocessa i dati usando la classe SeasonSentimentAnalysis
        sentiment_analysis = SeasonSentimentAnalysis(self.df)
        df_preprocessed = sentiment_analysis.preprocess()

        # Calcola il sentiment medio per stagione e per hotel, il punteggio medio e il delta
        seasonal_sentiment = df_preprocessed.groupBy("Hotel_Name","Season").agg(
            avg("Net_Sentiment").alias("avg_sentiment"),
            avg("Reviewer_Score").alias("avg_reviewer_score"),
            count("*").alias("review_count")
        ).withColumn(
            "sentiment_reviewer_delta",
            (col("avg_sentiment") - (col("avg_reviewer_score"))/10) #normalizziamo perchè altrimenti sarà sempre negativo, chi fantasia cu si query ma non potevamo fare una cosa ridicola???
        ).orderBy("Hotel_Name", "Season")

        # Mostra i risultati
        print("\nSentiment medio per stagione:")
        seasonal_sentiment.show(n, truncate=False)