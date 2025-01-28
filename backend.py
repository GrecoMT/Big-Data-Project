from pyspark.sql import SparkSession
from pyspark import SparkConf

from pyspark.sql.types import IntegerType, FloatType, StringType, BooleanType

from pyspark.sql.functions import (regexp_replace, split, expr, col, to_date, regexp_extract, udf, count, avg, first, explode, abs, desc, stddev, coalesce, to_date, when, date_format, datediff, row_number, lower) #ricordare di rimuovere gli import non usati

import utils

#update 28gennaio: cablaggio di un query manager all'interno dello spark builder
#from query_manager import QueryManager

from pyspark.sql.window import Window
from sklearn.cluster import DBSCAN

from coherence_review_model_vader import CoherenceReviewModel
from season_sentiment_analysis import SeasonSentimentAnalysis

import folium

import nltk
from nltk.corpus import wordnet


class SparkBuilder:
    def __init__(self, appname: str, dataset_path: str):
        
        conf = SparkConf() \
            .set("spark.driver.memory", "8g") \
            .set("spark.executor.memory", "8g")\
            .set('spark.executor.cores', "4")\
            .set('spark.driver.maxResultSize', "4gb")
            
        self.spark = (SparkSession.builder 
                    .config(conf=conf)    
                    .master("local[*]") 
                    .appName(appname)
                    .getOrCreate())
        
        # Imposto il log level su ERROR per evitare i warning
        self.spark.sparkContext.setLogLevel("ERROR")
        
        self.dataset = self.spark.read.csv(dataset_path, header=True, inferSchema=True)   
        self.castDataset()
        self.cleanDataset()
        self.df_finale = self.fillLatLng()

        #Query Manager associato alla sessione Spark
        self.queryManager = QueryManager(self)

    def get_spark_session(self):
        return self.spark      

    def castDataset(self):
        df = self.dataset
        
        #Cast delle colonne
        df = df.withColumn("Additional_Number_of_Scoring", df["Additional_Number_of_Scoring"].cast(IntegerType()))
        df = df.withColumn("Review_Date", to_date(df["Review_Date"], "M/d/yyyy"))
        df = df.withColumn("Average_Score", df["Average_Score"].cast(FloatType()))
        df = df.withColumn("Review_Total_Negative_Word_Counts", df["Review_Total_Negative_Word_Counts"] .cast(IntegerType()))
        df = df.withColumn("Total_Number_of_Reviews", df["Total_Number_of_Reviews"].cast(IntegerType()))
        df = df.withColumn("Review_Total_Positive_Word_Counts", df["Review_Total_Positive_Word_Counts"].cast(IntegerType()))
        df = df.withColumn("Total_Number_of_Reviews_Reviewer_Has_Given", df["Total_Number_of_Reviews_Reviewer_Has_Given"].cast(IntegerType()))
        df = df.withColumn("Reviewer_Score", df["Reviewer_Score"].cast(FloatType()))
        df = df.withColumn("days_since_review", regexp_extract(col("days_since_review"), r"(\d+)", 1).cast(IntegerType()))

        df = df.withColumn("lat", df["lat"].cast(FloatType()))
        df = df.withColumn("lng", df["lng"].cast(FloatType()))

        #Conversione colonna "Tags" in un array di Stringhe
        df = df.withColumn("Tags", regexp_replace(col("Tags"), r"[\[\]']", ""))
        df = df.withColumn("Tags", split(col("Tags"), ", "))
        df = df.withColumn("Tags", expr("transform(Tags, x -> trim(x))"))

        self.dataset = df
    
    '''def contaNulli(self):
        df = self.dataset
        # Conta i valori non nulli per ogni colonna
        for column in df.columns:
            non_null_count = df.filter(col(column).isNotNull()).count()
            total_count = df.count()
            missing_count = total_count - non_null_count
            print(f"Colonna: {column}, Non Null: {non_null_count}, Mancanti: {missing_count}")'''
    
    def checkStrings(self):
        df = self.dataset

        string_columns = [field.name for field in df.schema.fields if field.dataType.simpleString() == "string"]
        
        # Crea un dizionario per salvare i risultati
        empty_counts = {}

        # Controlla ogni colonna di tipo stringa
        for column in string_columns:
            count_empty = df.filter(
                (col(column).isNull()) | (col(column) == " ")
            ).count()
            empty_counts[column] = count_empty
        
        return empty_counts
    
    def cleanDataset(self):
        #In seguito all'esecuzione di checkStrings()
        #Elimina le 523 righe in cui la nazionalità del recensore è una stringa vuota
        df = self.dataset
        #df = df.filter(col("lat").isNotNull() & col("lat").isNotNull() & (col("Reviewer_Nationality") != " "))
        self.dataset = df.filter(col("Reviewer_Nationality") != " ").cache()
    

    def fillLatLng(self):
        
        #Estraggo solo gli hotel sprovvisti di lat e lng
        df_reqfill = self.dataset.filter(col("lat").isNull() | col("lng").isNull()).select("hotel_address", "lat", "lng").distinct() #17 righe DA RIEMPIRE.
        #Definisco UDF
        get_lat_udf = udf(utils.get_lat, FloatType())
        get_lng_udf = udf(utils.get_lng, FloatType())
        
        #Riempio le 17 righe
        df_reqfill = df_reqfill \
        .withColumn("lat", get_lat_udf(col("Hotel_Address"))) \
        .withColumn("lng", get_lng_udf(col("Hotel_Address")))
        #Rename per comodità (drop successivo più facile)
        df_reqfill = df_reqfill.withColumnRenamed("lat", "lat_f").withColumnRenamed("lng", "lng_f")
        #Merge
        df_merged = self.dataset.join(df_reqfill, on=["hotel_address"], how="left")
        # Usa coalesce per riempire i valori nulli in "lat" e "lng" di df con quelli di df1
        df_merged = df_merged.withColumn(
            "lat", 
            coalesce(self.dataset["lat"], df_reqfill["lat_f"])
        ).withColumn(
            "lng", 
            coalesce(self.dataset["lng"], df_reqfill["lng_f"])
        )
        # rimuovere le colonne duplicate di df1 ("lat" e "lng" di df1)
        ret = df_merged.drop("lat_f", "lng_f")
        return ret
    
def is_adjective_or_adverb(word):
    """
    Determina se una parola è un aggettivo (a) o un avverbio (r) utilizzando WordNet.
    """
    synsets = wordnet.synsets(word)
    if not synsets:
        return False
    return any(s.pos() == 'a' or s.pos()=='r' for s in synsets)


class QueryManager:
    def __init__(self, spark_builder: SparkBuilder):
        self.spark = spark_builder.spark

    #------------------------------QUERY 1----------------------------------------------
    def words_score_analysis(self, n=20, min_frequency=1000):
        """
        Analizza quali parole aggettivi nelle recensioni positive o negative sono indicatrici di punteggi alti o bassi,
        considerando solo parole con una certa frequenza minima.

        Args:
            n (int): Numero di parole da mostrare nelle classifiche.
            min_frequency (int): Frequenza minima di occorrenza delle parole per essere considerate.
        """
        is_adjective_udf = udf(is_adjective_or_adverb, BooleanType())
        df = self.spark.df_finale
        
        # Tokenizza e filtra le parole aggettivi nelle recensioni positive
        positive_words = df.select(
            col("Reviewer_Score"),
            explode(split(col("Positive_Review"), r"\s+")).alias("word")
        ).filter(col("word") != "")  # Rimuove parole vuote
        
        positive_words = positive_words.withColumn("word", lower(col("word")))

        positive_words_filtered = positive_words.filter(is_adjective_udf(col("word")))

        positive_word_scores = positive_words_filtered.groupBy("word") \
            .agg(
                avg("Reviewer_Score").alias("avg_score"),
                count("word").alias("word_count")
            ) \
            .filter(col("word_count") >= min_frequency) \
            .orderBy(desc("avg_score"))

        print(f"\nTop {n} aggettivi positivi con il punteggio più alto (min. {min_frequency} occorrenze):")
        positive_word_scores.select("word", "avg_score", "word_count").show(n, truncate=False)

        # Tokenizza e filtra le parole aggettivi nelle recensioni negative
        negative_words = df.select(
            col("Reviewer_Score"),
            explode(split(col("Negative_Review"), r"\s+")).alias("word")
        ).filter(col("word") != "")  # Rimuove parole vuote
        
        negative_words = negative_words.withColumn("word", lower(col("word")))

        negative_words_filtered = negative_words.filter(is_adjective_udf(col("word")))

        negative_word_scores = negative_words_filtered.groupBy("word") \
            .agg(
                avg("Reviewer_Score").alias("avg_score"),
                count("word").alias("word_count")
            ) \
            .filter(col("word_count") >= min_frequency) \
            .orderBy("avg_score")  # Ordina dal punteggio più basso

        print(f"\nTop {n} aggettivi negativi con il punteggio più basso (min. {min_frequency} occorrenze):")
        negative_word_scores.select("word", "avg_score", "word_count").show(n, truncate=False)

        return positive_word_scores, negative_word_scores
        
        
    # ------------------------------QUERY 2----------------------------------------------
    #Correlazione tra nazionalità e recensioni positive e negative.
    #Correlazione tra nazionalità e recensioni positive e negative.
    def nationality_review_analysis(self, n=20, min_reviews=2):
        """
        Analizza le nazionalità in base ai punteggi medi e alle recensioni positive/negative,
        considerando solo le nazionalità con un numero minimo di recensioni.

        Args:
            n (int): Numero di nazionalità da mostrare nelle classifiche.
            min_reviews (int): Numero minimo di recensioni per includere una nazionalità.
        """
        df = self.spark.df_finale

        # Nazionalità più "buone" (punteggi medi più alti)
        good_nationalities = df.groupBy("Reviewer_Nationality") \
            .agg(
                avg("Reviewer_Score").alias("avg_score"),
                count("*").alias("review_count")
            ).filter(col("review_count") >= min_reviews) \
            .orderBy(desc("avg_score"))

        print(f"\nTop {n} nazionalità con il punteggio medio più alto (min. {min_reviews} recensioni):")
        good_nationalities.show(n, truncate=False)

        # Nazionalità più "cattive" (punteggi medi più bassi)
        bad_nationalities = df.groupBy("Reviewer_Nationality") \
            .agg(
                avg("Reviewer_Score").alias("avg_score"),
                count("*").alias("review_count")
            ).filter(F.col("review_count") >= min_reviews) \
            .orderBy("avg_score")

        print(f"\nTop {n} nazionalità con il punteggio medio più basso (min. {min_reviews} recensioni):")
        bad_nationalities.show(n, truncate=False)

        # Correlazione tra nazionalità e lunghezza delle recensioni positive/negative
        nationality_reviews = df.groupBy("Reviewer_Nationality") \
            .agg(
                avg("Review_Total_Positive_Word_Counts").alias("avg_positive_words"),
                avg("Review_Total_Negative_Word_Counts").alias("avg_negative_words"),
                count("*").alias("review_count")
            ).filter(col("review_count") >= min_reviews) \
            .orderBy(desc("avg_positive_words"))

        print(f"\nCorrelazione tra nazionalità e lunghezza delle recensioni positive/negative (min. {min_reviews} recensioni):")
        nationality_reviews.select(
            "Reviewer_Nationality", "avg_positive_words", "avg_negative_words", "review_count"
        ).show(n, truncate=False)

        return nationality_reviews
        
    # ------------------------------QUERY 3----------------------------------------------

  #Influenza delle tag sullo scoring
    def tag_influence_analysis(self, n=10, min_count=100):
        """
        Analizza l'influenza delle tag sul punteggio dato, considerando solo i tag usati un numero minimo di volte.

        Args:
            n (int): Numero di tag da mostrare nelle classifiche.
            min_count (int): Conteggio minimo per includere un tag nell'analisi.
        """
        df = self.spark.df_finale

        # Esplodi la colonna Tags in righe individuali
        exploded_tags = df.select(
            col("Reviewer_Score"),
            explode(col("Tags")).alias("tag")
        )

        # Calcola la media del punteggio e il conteggio per ciascuna tag
        tag_scores = exploded_tags.groupBy("tag") \
            .agg(
                avg("Reviewer_Score").alias("avg_score"),
                count("*").alias("tag_count")
            ) \
            .filter(col("tag_count") >= min_count).orderBy(desc("avg_score"))

        print(f"\nTop {n} tag con il punteggio medio più alto (min. {min_count} utilizzi):")
        tag_scores.show(n, truncate=False)

        # Tag con punteggio più basso
        print(f"\nTop {n} tag con il punteggio medio più basso (min. {min_count} utilizzi):")
        tag_scores.orderBy("avg_score").show(n, truncate=False)

        # Filtra per specifiche tag di interesse
        specific_tags = ["Solo traveler", "Business trip", "Leisure trip", "Couple"]
        filtered_tags = tag_scores.filter(col("tag").isin(specific_tags))

        print("\nInfluenza delle tag selezionate:")
        filtered_tags.show(truncate=False)
        return filtered_tags
    
    # ------------------------------QUERY 4----------------------------------------------
    # Analisi degli outlier: identificare discrepanze significative tra il punteggio medio di un hotel e il numero di recensioni positive e negative.
    def outlier_analysis(self, score_threshold_high=8.5, score_threshold_low=5.0, review_threshold=10):
        """
        Identifica hotel con recensioni anomale:
        - Punteggio alto, ma molte recensioni negative.
        - Punteggio basso, ma molte recensioni positive.
        """
        df = self.spark.df_finale

        # Raggruppa per hotel e calcola metriche
        hotel_reviews = df.groupBy("Hotel_Address", "Hotel_Name") \
            .agg(
                avg("Reviewer_Score").alias("avg_score"),
                sum(when(col("Negative_Review") != "No Negative", 1).otherwise(0)).alias("total_negative_reviews"),
                sum(when(col("Positive_Review") != "No Positive", 1).otherwise(0)).alias("total_positive_reviews"),
                count("*").alias("review_count")
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
        
        #CAPIRE COSA RITORNARE ORA NON HO TEMPO

    # ------------------------------QUERY 5----------------------------------------------
    #Analisi della lunghezza: verificare se recensioni più lunghe tendono ad essere più positive o negative.
    def review_length_analysis(self, n=20):
        """
        Analizza la lunghezza delle recensioni e verifica se recensioni più lunghe 
        tendono ad essere più positive o negative.
        """
        df = self.spark.df_finale
        df.filter((col("Review_Total_Positive_Word_Counts") > 0) | (col("Review_Total_Negative_Word_Counts") > 0)) #aggiunto per ottimizzare, filtro le recensioni vuote
        
        # Calcolo della lunghezza media delle recensioni positive e negative per punteggio
        review_length = df.groupBy("Reviewer_Score") \
            .agg(
                avg("Review_Total_Positive_Word_Counts").alias("avg_positive_length"),
                avg("Review_Total_Negative_Word_Counts").alias("avg_negative_length"),
                count("*").alias("review_count")
            ).orderBy("Reviewer_Score")

        print("\nLunghezza media delle recensioni positive e negative per punteggio:")
        review_length.show(n, truncate=False)

        # Analisi generale: recensioni più lunghe tendono ad essere più positive o negative?
        avg_positive = df.agg(avg("Review_Total_Positive_Word_Counts").alias("overall_avg_positive_length")).collect()[0][0]
        avg_negative = df.agg(avg("Review_Total_Negative_Word_Counts").alias("overall_avg_negative_length")).collect()[0][0]

        print(f"\nLunghezza media generale delle recensioni positive: {avg_positive:.2f}")
        print(f"Lunghezza media generale delle recensioni negative: {avg_negative:.2f}")
        
        return df
        
#--------------------------QUERY 6------------------------------------------------
#Analisi della coerenza tra recensioni e punteggi:predire il punteggio basato sul contenuto della recensione e confrontarlo con il punteggio effettivamente dato
    def coherence_analysis(self, threshold=2.0,n=10, export_path=None):
        """
        Richiama l'analisi del modello di sentiment e coerenza.
        """
        sentiment_model = CoherenceReviewModel(self.spark.df_finale)

        # Preprocessa il testo
        sentiment_model.preprocess_reviews()

        # Addestra il modello
        sentiment_model.train_sentiment_model()

        # Esegui l'analisi di coerenza
        predictions = sentiment_model.analyze_consistency(threshold,n, export_path) # treshold serve a defnire un limite massimo accettabile per l'errore assoluto tra il punteggio predetto dal modello e il punteggio reale
        
        return predictions
        
        
#--------------------------QUERY 7------------------------------------------------
#Durata della percezione negativa: analizzare se la percezione negativa diminuisce nel tempo.
    def recovery_time_analysis(self, n=20):
        """
        Analizza quanto tempo ci vuole affinché un hotel passi da una recensione negativa (<6)
        a una recensione positiva (>6). Calcola medie per ogni fase e verifica il trend futuro.
        """
        df = self.spark.df_finale

        # Ordina le recensioni per hotel e data
        #window_spec = Window.partitionBy("Hotel_Name").orderBy("Review_Date")

        # Finestra di 10 recensioni precedenti per calcolare la media
        window_prev = Window.partitionBy("Hotel_Name").orderBy("Review_Date").rowsBetween(-10, 0)

        # Finestra di 5 recensioni successive per il trend dopo la positiva
        window_after_positive = Window.partitionBy("Hotel_Name").orderBy("Review_Date").rowsBetween(1, 5)

        # Calcolo di avg_after_negative (considerando tutte le recensioni precedenti)
        df = df.withColumn("avg_after_negative", avg("Reviewer_Score").over(window_prev))

        # Filtra recensioni negative
        df_negative = df.filter(col("Reviewer_Score") < 6).withColumnRenamed(
            "Reviewer_Score", "Negative_Score"
        ).withColumnRenamed("Review_Date", "Negative_Review_Date").withColumnRenamed(
            "avg_after_negative", "avg_after_negative_score"
        )

        # Filtra recensioni positive
        df_positive = df.filter(col("Reviewer_Score") > 6).withColumn(
            "avg_trend_after_positive",
            avg("Reviewer_Score").over(window_after_positive)
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
            datediff(col("Positive_Review_Date"), col("Negative_Review_Date"))
        ).filter(col("days_between") > 0)  # Escludi casi senza recupero

        # Calcolo di avg_after_positive
        df_recovery = df_recovery.withColumn(
            "avg_after_positive",
            (col("avg_after_negative_score") + col("Positive_Score")) / 2
        )

        # Filtro per il days_between minimo per ogni hotel
        window_min_days = Window.partitionBy("Hotel_Name").orderBy("days_between")
        df_recovery_filtered = df_recovery.withColumn(
            "rank", row_number().over(window_min_days)
        ).filter(col("rank") == 1)  # Mantieni solo la riga con il days_between minimo

        # Calcola il tempo medio di recupero
        avg_recovery_time = df_recovery_filtered.agg(avg("days_between").alias("avg_recovery_time")).collect()[0][0]

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
        
        return result
        
        
#--------------------------QUERY 8------------------------------------------------

    #Analisi della reputazione: differenza tra il punteggio medio storico di un hotel e il punteggio medio delle recensioni recenti
    def reputation_analysis(self, recent_reviews=30, n=20, score_difference = -1):
        """
        Analizza la differenza tra il punteggio medio storico di un hotel e il punteggio medio delle recensioni recenti.
        """
        df = self.spark.df_finale

        # Calcolo del punteggio medio storico per ogni hotel
        avg_historical_window = Window.partitionBy("Hotel_Name")
        df = df.withColumn(
            "avg_historical_score",
            avg("Reviewer_Score").over(avg_historical_window)
        )

        # Calcolo del punteggio medio delle recensioni recenti (finestra di N recensioni)
        recent_reviews_window = Window.partitionBy("Hotel_Name").orderBy(desc("Review_Date")).rowsBetween(Window.currentRow, Window.currentRow + recent_reviews - 1)

        df = df.withColumn(
            "avg_recent_score",
            avg("Reviewer_Score").over(recent_reviews_window)
        )

        # Filtro per ottenere una riga per ogni hotel
        df_aggregated = df.groupBy("Hotel_Name").agg(
            first("avg_historical_score").alias("avg_historical_score"),
            first("avg_recent_score").alias("avg_recent_score")
        )

        # Calcolo della differenza tra il punteggio recente e quello storico
        df_aggregated = df_aggregated.withColumn(
            "score_difference",
            col("avg_recent_score") - col("avg_historical_score")
        )

        # Risultati finali
        print("\nRisultati dell'analisi della reputazione:")
        df_aggregated.orderBy("score_difference", ascending=False).filter(col("score_difference")<score_difference).show(n, truncate=False)

        # Media delle differenze
        avg_difference = df_aggregated.agg(avg("score_difference").alias("avg_difference")).collect()[0][0]
        print(f"\nDifferenza media complessiva: {avg_difference:.2f}")
        
        return df_aggregated
                
#-----------------------QUERY 9------------------------------------------------
    def seasonal_sentiment_analysis(self, n=4):
        """
        Analizza come il sentiment medio delle recensioni varia in base alla stagione dell'anno.
        """
        # Preprocessa i dati usando la classe SeasonSentimentAnalysis
        sentiment_analysis = SeasonSentimentAnalysis(self.spark.df_finale)
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
        
        return seasonal_sentiment
        
#-----------------------QUERY 10------------------------------------------------
    #dentificare recensioni sospette tipo recensioni estremamente positive o negative che differiscono significativamente dal trend generale dell’hotel
    def anomaly_detection(self,n=20):
        
        df = self.spark.df_finale

        # Calcola la media e deviazione standard del punteggio per ogni hotel
        hotel_stats = df.groupBy("Hotel_Name").agg(
            avg("Reviewer_Score").alias("Avg_Score"),
            stddev("Reviewer_Score").alias("Std_Dev_Score")
        )

        # Unisci il trend generale al dataframe originale
        df_with_stats = df.join(hotel_stats, on="Hotel_Name")

        # Identifica recensioni estremamente positive o negative rispetto al trend
        extreme_reviews = df_with_stats.filter(
            abs(col("Reviewer_Score") - col("Avg_Score")) > (2 * col("Std_Dev_Score"))
        )

        # Visualizza le recensioni anomale
        extreme_reviews.select(
            "Hotel_Name", "Reviewer_Score", "Avg_Score", "Positive_Review", "Negative_Review", "Reviewer_Score",
        ).show(n, truncate = True)
        
        return extreme_reviews
        
#-----------------------QUERY 11------------------------------------------------
    def location_influence(self, n=20):
        
        df = self.spark.df_finale
        
        # Filtra le colonne necessarie
        df_geo = df.select("Hotel_Name", "Average_Score", "lat", "lng").distinct().dropna() 
        df_geo.show()
        
        # Converti in Pandas per DBSCAN
        sample_df = df_geo.limit(10000)
        sample_pandas = sample_df.toPandas() #faccio così senno muore il processo

        #df_geo_pandas = df_geo.toPandas()

        # Prepara i dati per il clustering
        coords = sample_pandas[["lat", "lng"]].to_numpy()

        # Applica DBSCAN
        dbscan = DBSCAN(eps=0.05, min_samples=5, metric="euclidean")
        clusters = dbscan.fit_predict(coords)

        # Aggiungi i cluster al DataFrame Pandas
        sample_pandas["Cluster"] = clusters

        # Ritorna i dati in Spark
        df_clustered = self.spark.createDataFrame(sample_pandas)

        # Analizza i cluster in Spark
        # Calcola la posizione geografica media e il punteggio medio per ogni cluster
        cluster_summary = df_clustered.groupBy("Cluster").agg(
            avg("lat").alias("Avg_Lat"),
            avg("lng").alias("Avg_Lng"),
            avg("Average_Score").alias("Avg_Score"),
            count("Hotel_Name").alias("Hotel_Count"),
            first("Hotel_Name").alias("Example_Hotel")
       )
     
        cluster_summary.show() #Lo vedo perche non sto capendo che caspita si prende
        
        # Filtra i dati per il cluster desiderato
        cluster_data = sample_pandas[sample_pandas["Cluster"] == 0]

        # Calcola centro e raggio del cluster
        cluster_center_lat, cluster_center_lng = utils.calculate_cluster_center(cluster_data)
        cluster_radius_km = utils.calculate_cluster_radius(cluster_data, cluster_center_lat, cluster_center_lng)

        # Crea la mappa
        m = folium.Map(location=[45.4654, 9.1859], zoom_start=6)
        
        # Itera su ogni cluster
        for cluster_id in sample_pandas["Cluster"].unique():
            if cluster_id == -1:  # Ignora i punti rumore
                continue

            # Filtra i dati del cluster corrente
            cluster_data = sample_pandas[sample_pandas["Cluster"] == cluster_id]

            # Calcola il centro e il raggio del cluster
            cluster_center_lat, cluster_center_lng = utils.calculate_cluster_center(cluster_data)
            cluster_radius_km = utils.calculate_cluster_radius(cluster_data, cluster_center_lat, cluster_center_lng)

            # Aggiungi un cerchio per il cluster corrente
            folium.Circle(
                location=[cluster_center_lat, cluster_center_lng],
                radius=cluster_radius_km * 1000,  # Converti il raggio in metri
                color="blue",
                fill=True,
                fill_opacity=0.5,
                popup=f"Cluster: {cluster_id}, Avg Rating: {cluster_data['Average_Score'].mean():.2f}"
            ).add_to(m)

        # Salva o mostra la mappa
        #m.show_in_browser()
        return m

#--------------- COUNT RECENSIONI POS-NEG PER SEASON ---------------#
    def prova_stagione(self):
        # UDF per calcolare la stagione
        season_udf = udf(lambda date: utils.get_season(date.month), StringType())

        # Aggiungere colonna stagione
        #df_season = self.spark.df_finale.withColumn("season", season_udf(to_date("Review_Date", "yyyy-MM-dd")))
        df_season = self.spark.df_finale.withColumn("season", season_udf(col("Review_Date")))

        # Calcolare se una recensione è positiva o negativa
        '''df_season = df_season.withColumn("is_positive", when(col("Positive_Review") != "No Positive", 1).otherwise(0))
        df_season = df_season.withColumn("is_negative", when(col("Negative_Review") != "No Negative", 1).otherwise(0))'''

        # Raggruppare per Hotel_Name e stagione e calcolare direttamente i conteggi
        seasonal_counts = (
            df_season.groupBy("Hotel_Name", "season")
            .agg(
                sum(when(col("Positive_Review") != "No Positive", 1).otherwise(0)).alias("positive_reviews"),
                sum(when(col("Negative_Review") != "No Negative", 1).otherwise(0)).alias("negative_reviews"),
            )
        )

        seasonal_counts.orderBy("hotel_name", "season").show()
    
        # Pivot per avere stagioni come colonne
        '''pivot_result = seasonal_counts.groupBy("Hotel_Name").pivot("season").agg(
            first("positive_reviews").alias("Pos"),
            first("negative_reviews").alias("Neg")
        )

        # Ordinare le colonne per maggiore chiarezza
        final_result = pivot_result.select(
            "Hotel_Name",
            "Winter_Pos", "Winter_Neg",
            "Spring_Pos", "Spring_Neg",
            "Summer_Pos", "Summer_Neg",
            "Autumn_Pos", "Autumn_Neg"
        )

        # Mostrare il risultato
        final_result.show()'''

#----------------- TREND MESE-ANNO  ------------------#
    def trend_mensile(self):
        #Creazione colonna "YearMonth" che contiene l'anno e il mese
        df_trend = self.spark.df_finale.withColumn("YearMonth", date_format(col("Review_Date"), "yyyy-MM"))
        
        # Aggregare per "Hotel_Name" e "YearMonth" e calcolare la media degli score
        trend_df = df_trend.groupBy("Hotel_Name", "YearMonth").agg(
            avg("Reviewer_Score").alias("Average_Score")
        ).orderBy("Hotel_Name", "YearMonth")

        #provoGrafico
        #trend_df = trend_df.filter(col("Hotel_Name")=="11 Cadogan Gardens")
        utils.graficoTrend(trend_df)
        
        # Mostrare i dati aggregati
        #trend_df.show(50)
        #Creazione colonna "YearMonth" che contiene l'anno e il mese
        df_trend = self.df.withColumn("YearMonth", date_format(col("Review_Date"), "yyyy-MM"))
        
        # Aggregare per "Hotel_Name" e "YearMonth" e calcolare la media degli score
        trend_df = df_trend.groupBy("Hotel_Name", "YearMonth").agg(
            avg("Reviewer_Score").alias("Average_Score")
        ).orderBy("Hotel_Name", "YearMonth")

        #provoGrafico
        #trend_df = trend_df.filter(col("Hotel_Name")=="11 Cadogan Gardens")
        utils.graficoTrend(trend_df)
        
        # Mostrare i dati aggregati
        #trend_df.show(50)