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

    #Query 1
    def words_score_analysis(self, min_frequency=1000):
        is_adjective_or_adverb_udf = udf(utils.is_adjective_or_adverb, BooleanType())
        df = self.spark.df_finale
    
        #Aggettivi e avverbi nelle recensioni positive
        positive_words = df.select(
            col("Reviewer_Score"),
            explode(split(col("Positive_Review"), r"\s+")).alias("word")
        ).filter(col("word") != "")  # Rimuove parole vuote
        positive_words = positive_words.withColumn("word", lower(col("word")))
        positive_words_filtered = positive_words.filter(is_adjective_or_adverb_udf(col("word")))
        positive_word_scores = positive_words_filtered.groupBy("word") \
            .agg(
                avg("Reviewer_Score").alias("avg_score"),
                count("word").alias("word_count")
            ) \
            .filter(col("word_count") >= min_frequency) \
            .orderBy(desc("avg_score"))
        #Aggettivi e avverbi nelle recensioni negative
        negative_words = df.select(
            col("Reviewer_Score"),
            explode(split(col("Negative_Review"), r"\s+")).alias("word")
        ).filter(col("word") != "")  # Rimuove parole vuote
        negative_words = negative_words.withColumn("word", lower(col("word")))
        negative_words_filtered = negative_words.filter(is_adjective_or_adverb_udf(col("word")))
        negative_word_scores = negative_words_filtered.groupBy("word") \
            .agg(
                avg("Reviewer_Score").alias("avg_score"),
                count("word").alias("word_count")
            ) \
            .filter(col("word_count") >= min_frequency) \
            .orderBy("avg_score")  # Ordina dal punteggio più basso
        return positive_word_scores, negative_word_scores
        
    #Query 2
    def tag_influence_analysis(self, min_count=1000):
        df = self.spark.df_finale

        # Esplodi la colonna Tags in righe individuali
        exploded_tags = df.select(
            col("Reviewer_Score"),
            explode(col("Tags")).alias("tag")
        )
        # Calcola la media del punteggio e il conteggio per ciascun tag
        tag_scores_desc = exploded_tags.groupBy("tag") \
            .agg(
                avg("Reviewer_Score").alias("avg_score"),
                count("*").alias("tag_count")
            ) \
            .filter(col("tag_count") >= min_count).orderBy(desc("avg_score"))

        return tag_scores_desc
    
    #Query 3
    def review_length_analysis(self):
        df = self.spark.df_finale.filter((col("Review_Total_Positive_Word_Counts") > 0) | (col("Review_Total_Negative_Word_Counts") > 0))
        
        # Calcolo della lunghezza media delle recensioni positive e negative per punteggio
        review_length = df.groupBy("Reviewer_Score") \
            .agg(
                avg("Review_Total_Positive_Word_Counts").alias("avg_positive_length"),
                avg("Review_Total_Negative_Word_Counts").alias("avg_negative_length")
            ).orderBy("Reviewer_Score")
        
        return review_length
    
    #Query 4
    def reputation_analysis(self, recent_reviews=30, score_difference = -1):
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

        # Media delle differenze
        #avg_difference = df_aggregated.agg(avg("score_difference").alias("avg_difference")).collect()[0][0]
        
        return df_aggregated
    #Query 4 singola
    def reputation_analysis_single(self, hotel_name, recent_reviews=30):
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
                
    #Query 5
    def seasonal_sentiment_analysis(self):
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
        
    #Query 6
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
    
    #Query 7
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

    #Query 8
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
                first("lng").alias("Longitude")
            )
            return res

    #Roberta's Functions (9)
    def analyze_hotel_sentiment(self, hotel_name):
        """Calcola il sentiment medio delle recensioni di un hotel."""
        print(f"Analizzando il sentiment di {hotel_name}")
        return self.sentiment_analyzer.analyze_hotel_sentiment(hotel_name, self.spark.df_finale) 

    #DeepSeek Sum (10)
    def sumReviews(self, hotel_name):
        return self.summaryLLM.getSummary(hotel_name)

    #Query 11
    def get_hotels_by_tag(self, city, tag):
        df_ret = self.spark.df_finale.filter(col("Hotel_Address").contains(city) & array_contains(col("Tags"),tag)).groupBy("Hotel_Name", "Hotel_Address").agg(
            avg("Reviewer_Score").alias("avg_score") 
        ).orderBy(desc("avg_score"))
        
        return df_ret
    
    #Query 12
    def get_nearby_hotels(self, user_lat, user_lng, max_distance):
        # Registra la funzione come UDF
        haversine_udf = udf(lambda lat, lng, user_lat, user_lng: utils.haversine(lat, lng, user_lat, user_lng))
        df_hotels = self.spark.df_finale
        return df_hotels.withColumn(
            "distance", haversine_udf(col("lat"), col("lng"), lit(user_lat), lit(user_lng))
        ).filter(col("distance") <= max_distance)
    
    #Query 13
    def trend_mensile_compare(self, dataframe):
        
        #Creazione colonna "YearMonth" che contiene l'anno e il mese
        df_trend = dataframe.withColumn("YearMonth", date_format(col("Review_Date"), "yyyy-MM"))
        
        # Aggregare per "Hotel_Name" e "YearMonth" e calcolare la media degli score
        trend_df = df_trend.groupBy("Hotel_Name", "YearMonth").agg(
            avg("Reviewer_Score").alias("Average_Score")
        ).orderBy("Hotel_Name", "YearMonth")

        return utils.graficoTrend(trend_df, False)  

    #Query 14
    def get_most_used_tags(self):
        # Esplodere l'array "tags" in valori singoli
        tags_df = self.spark.df_finale.select(explode(col("tags")).alias("tag"))
        # Contare la frequenza di ogni tag e ordinarli in ordine decrescente
        tags_count_df = tags_df.groupBy("tag").agg(count("*").alias("count")).orderBy(desc("count"))
        tags = tags_count_df.select("tag")
        
        return tags
    
    #Query 15
    def nationality_review_analysis(self, min_reviews):

        df = self.spark.df_finale
        nationality_reviews = df.groupBy("Reviewer_Nationality") \
            .agg(
                count("*").alias("review_count")
            ).filter(col("review_count") >= min_reviews)
        return nationality_reviews
    
    #Query 16
    def get_reviews_for_nationality(self, nationality):
        # Filtra i dati per la nazionalità selezionata
        return self.spark.df_finale.filter(col("Reviewer_Nationality") == nationality).select("Hotel_Name", "Review_Date", "Reviewer_Nationality", "Reviewer_Score","Negative_Review", "Review_Total_Negative_Word_Counts", "Positive_Review", "Review_Total_Positive_Word_Counts",  "Tags", )
    
    

