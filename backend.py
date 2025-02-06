from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType, FloatType, BooleanType
from pyspark.sql.functions import regexp_replace, split, expr, col, to_date, regexp_extract, udf, count, array_contains, avg, first, explode, abs, desc, asc, stddev, coalesce, to_date, when, date_format, lower, lit, sum, max, min
import utils
from pyspark.sql.window import Window
from season_sentiment_analysis import SeasonSentimentAnalysis
from RoBERTa_Sentiment import RoBERTa_Sentiment
from DeepSeekSum import SummaryLLM

dataset_path = "/Users/vincenzopresta/Desktop/Big Data/dataset/Hotel_Reviews.csv"
#dataset_path = "/Users/matteog/Documents/Università/Laurea Magistrale/Big Data/Progetto/Dataset/Hotel_Reviews.csv"

class SparkBuilder:
    def __init__(self, appname: str):            
        self.spark = (SparkSession.builder 
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


class QueryManager:
    def __init__(self, spark_builder: SparkBuilder):
        
        self.spark = spark_builder
        self.sentiment_analyzer = RoBERTa_Sentiment()
        self.summaryLLM = SummaryLLM(self.spark.df_finale)

    #------------------------------QUERY 1----------------------------------------------
    def words_score_analysis(self, n=20, min_frequency=1000):
        """
        Analizza quali parole aggettivi nelle recensioni positive o negative sono indicatrici di punteggi alti o bassi,
        considerando solo parole con una certa frequenza minima.

        Args:
            n (int): Numero di parole da mostrare nelle classifiche.
            min_frequency (int): Frequenza minima di occorrenza delle parole per essere considerate.
        """
        is_adjective_udf = udf(utils.is_adjective_or_adverb, BooleanType())
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
            ).filter(col("review_count") >= min_reviews) \
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
    def tag_influence_analysis(self, min_count=1000):
        df = self.spark.df_finale

        # Esplodi la colonna Tags in righe individuali
        exploded_tags = df.select(
            col("Reviewer_Score"),
            explode(col("Tags")).alias("tag")
        )

        # Calcola la media del punteggio e il conteggio per ciascuna tag
        tag_scores_desc = exploded_tags.groupBy("tag") \
            .agg(
                avg("Reviewer_Score").alias("avg_score"),
                count("*").alias("tag_count")
            ) \
            .filter(col("tag_count") >= min_count).orderBy(desc("avg_score"))

        tag_scores_desc.show(10)

        tag_scores_asc = tag_scores_desc.orderBy(asc("avg_score"))
        return tag_scores_asc
    
    # ------------------------------QUERY 4----------------------------------------------
    #Analisi della lunghezza: verificare se recensioni più lunghe tendono ad essere più positive o negative.
    def review_length_analysis(self):
        """
        Analizza la lunghezza delle recensioni e verifica se recensioni più lunghe 
        tendono ad essere più positive o negative.
        """
        df = self.spark.df_finale.filter((col("Review_Total_Positive_Word_Counts") > 0) | (col("Review_Total_Negative_Word_Counts") > 0)) #aggiunto per ottimizzare, filtro le recensioni vuote
        
        # Calcolo della lunghezza media delle recensioni positive e negative per punteggio
        review_length = df.groupBy("Reviewer_Score") \
            .agg(
                avg("Review_Total_Positive_Word_Counts").alias("avg_positive_length"),
                avg("Review_Total_Negative_Word_Counts").alias("avg_negative_length")
            ).orderBy("Reviewer_Score")
        
        return review_length

#--------------------------QUERY 5------------------------------------------------
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
        df_aggregated.orderBy("score_difference", ascending=False).filter(col("score_difference")<score_difference).show(n, truncate=False)

        # Media delle differenze
        avg_difference = df_aggregated.agg(avg("score_difference").alias("avg_difference")).collect()[0][0]
        
        return df_aggregated
    
    def reputation_analysis_single(self, hotel_name, recent_reviews=30):
        """
        Analizza la differenza tra il punteggio medio storico di un hotel e il punteggio medio delle recensioni recenti.
        """
        df = self.spark.df_finale.filter(col("Hotel_Name") == hotel_name)

        if df.count() == 0:
            print(f"Nessuna recensione trovata per l'hotel: {hotel_name}")
            return None

        # Calcolo del punteggio medio storico per l'hotel specificato
        avg_historical_window = Window.partitionBy("Hotel_Name")
        df = df.withColumn(
            "avg_historical_score", avg("Reviewer_Score").over(avg_historical_window)
        )

        # Calcolo del punteggio medio delle recensioni recenti (finestra di N recensioni)
        recent_reviews_window = Window.partitionBy("Hotel_Name").orderBy(desc("Review_Date")).rowsBetween(0, recent_reviews - 1)
        df = df.withColumn(
            "avg_recent_score", avg("Reviewer_Score").over(recent_reviews_window)
        )

        # Estrazione di una singola riga con i dati aggregati
        df_aggregated = df.groupBy("Hotel_Name").agg(
            first("avg_historical_score").alias("avg_historical_score"),
            first("avg_recent_score").alias("avg_recent_score")
        )

        # Calcolo della differenza tra il punteggio recente e quello storico
        df_aggregated = df_aggregated.withColumn(
            "score_difference", col("avg_recent_score") - col("avg_historical_score")
        )

        return df_aggregated
                
#--------------------------- QUERY 6------------------------------------------------
    #Analisi del sentiment in base alla stagione
    def seasonal_sentiment_analysis(self):
        """
        Analizza come il sentiment medio delle recensioni varia in base alla stagione dell'anno.
        Utilizza VADER -> rule based e lessico
        """
        sentiment_analysis = SeasonSentimentAnalysis(self.spark.df_finale)

        # Preprocessa i dati per calcolare il sentiment e la stagione
        df_preprocessed = sentiment_analysis.preprocess()

        # Aggrega il sentiment per hotel e stagione
        seasonal_sentiment = df_preprocessed.groupBy("Hotel_Name", "Hotel_Address", "Season").agg(
            avg("Net_Sentiment").alias("avg_sentiment"),
            avg("Reviewer_Score").alias("avg_reviewer_score"),
            count("*").alias("review_count")
        ).orderBy("Hotel_Name", "Season")

        return seasonal_sentiment
        
        
    #-----------------------QUERY 7------------------------------------------------
    #Identificare recensioni sospette tipo recensioni estremamente positive o negative che differiscono significativamente dal trend generale dell’hotel
    def anomaly_detection(self, hotelName):
        
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

        df_fin = extreme_reviews.filter(col("Hotel_Name")==hotelName).select(
             "Reviewer_Score", "Avg_Score", "Positive_Review", "Negative_Review"
        ).orderBy(asc("Reviewer_Score"))
        
        return df_fin
    
#----------------- QUERY 8 TREND MESE-ANNO  ------------------#
    def trend_mensile(self, hotelName):
        # Filtrare i dati per l'hotel specificato
        df_trend = self.spark.df_finale.filter(col("Hotel_Name") == hotelName)
        
        #Creazione colonna "YearMonth" che contiene l'anno e il mese
        df_trend = df_trend.withColumn("YearMonth", date_format(col("Review_Date"), "yyyy-MM"))
        
        # Aggregare per "Hotel_Name" e "YearMonth" e calcolare la media degli score
        trend_df = df_trend.groupBy("Hotel_Name", "YearMonth").agg(
            avg("Reviewer_Score").alias("Average_Score")
        ).orderBy("Hotel_Name", "YearMonth")

        return utils.graficoTrend(trend_df, True)

    #-------------- QUERY 9 STATISTICHE------------------#
    def hotelStatistics(self):
            df = self.spark.df_finale

            res = df.groupBy("Hotel_Name").agg(
                count("*").alias("Total_Reviews"),
                sum(when(col("Reviewer_Score") >= 6, 1).otherwise(0)).alias("Total_Positive_Reviews"),
                sum(when(col("Reviewer_Score") <  6, 1).otherwise(0)).alias("Total_Negative_Reviews"),
                max("Reviewer_Score").alias("Max_Reviewer_Score"),
                min("Reviewer_Score").alias("Min_Reviewer_Score"),
                avg("Reviewer_Score").alias("Avg_Reviewer_Score"),
                first("lat").alias("Latitude"),
                first("lng").alias("Longitude"),
                avg("Additional_Number_of_Scoring").alias("Avg_Additional_Number_of_Scoring")
            )
            return res

#------------------QUERY SUPPORTO 1 --------------------------------
    #in base al tag scelto restituire hotel che hanno quei tag con recensione più alta / bassa.

    def get_hotels_by_tag(self, city, tag):
        """
        city: Nome della città su cui filtrare gli hotel
        selected_tag: Tag selezionato per filtrare gli hotel
        order: "highest" per punteggio più alto, "lowest" per punteggio più basso
        """

        # Estrarre la città dall'indirizzo dell'hotel (ipotizzando che sia alla fine dell'indirizzo)
        df_nuovo = self.spark.df_finale.filter(col("Hotel_Address").contains(city) & array_contains(col("Tags"),tag)).groupBy("Hotel_Name", "Hotel_Address").agg(
            avg("Reviewer_Score").alias("avg_score") 
        ).orderBy(desc("avg_score"))
        
        return df_nuovo
    
    #------------QUERY SUPPORTO 2 : HOTEL VICINI----------------------------------------------------------------
    # Funzione per filtrare gli hotel
    def get_nearby_hotels(self, user_lat, user_lng, max_distance):
        # Registra la funzione come UDF
        haversine_udf = udf(lambda lat, lng, user_lat, user_lng: utils.haversine(lat, lng, user_lat, user_lng))
        df_hotels = self.spark.df_finale
        return df_hotels.withColumn(
            "distance", haversine_udf(col("lat"), col("lng"), lit(user_lat), lit(user_lng))
        ).filter(col("distance") <= max_distance)
    
    #-----------QUERY SUPPORTO 3--------
    def trend_mensile_compare(self, dataframe):
        
        #Creazione colonna "YearMonth" che contiene l'anno e il mese
        df_trend = dataframe.withColumn("YearMonth", date_format(col("Review_Date"), "yyyy-MM"))
        
        # Aggregare per "Hotel_Name" e "YearMonth" e calcolare la media degli score
        trend_df = df_trend.groupBy("Hotel_Name", "YearMonth").agg(
            avg("Reviewer_Score").alias("Average_Score")
        ).orderBy("Hotel_Name", "YearMonth")

        return utils.graficoTrend(trend_df, False)  

    #-----------QUERY SUPPORT 4------
    def get_most_used_tags(self):

        # Esplodere l'array "tags" in valori singoli
        tags_df = self.spark.df_finale.select(explode(col("tags")).alias("tag"))

        # Contare la frequenza di ogni tag e ordinarli in ordine decrescente
        tags_count_df = tags_df.groupBy("tag").agg(count("*").alias("count")).orderBy(desc("count"))
        
        tags = tags_count_df.select("tag")
        
        return tags
    
#----------------------------------------------------------------ROBERTA
    def analyze_single_review(self, positive_text, negative_text):
            """Analizza il sentiment di una singola recensione."""
            return self.sentiment_analyzer.analyze_review_sentiment(positive_text, negative_text) #questa funzione si può togliere se non integriamo l'analisi della singola recensione

    def analyze_hotel_sentiment(self, hotel_name):
        """Calcola il sentiment medio delle recensioni di un hotel."""
        print(f"Analizzando il sentiment di {hotel_name}")
        return self.sentiment_analyzer.analyze_hotel_sentiment(hotel_name, self.spark.df_finale) 

#----------------------------------------------------------------DeepSeekSum
    def sumReviews(self, hotel_name):
        return self.summaryLLM.getSummary(hotel_name)