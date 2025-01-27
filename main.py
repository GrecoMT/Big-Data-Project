from spark_builder import SparkBuilder
from query_manager import QueryManager
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower
import os 

def clear_terminal():
    os.system('cls' if os.name == 'nt' else 'clear')


if __name__ == '__main__':
    
    clear_terminal()
    
    dataset_path = "/Users/vincenzopresta/Desktop/Big Data/dataset/Hotel_Reviews.csv"
    spark_builder = SparkBuilder(appname="BigData_App", dataset_path=dataset_path)

    # Controlla se i dati sono puliti
    #spark_builder.df_finale.select([F.count(F.when(F.col(c).isNull(), 1)).alias(c) for c in spark_builder.df_finale.columns]).show()
    #spark_builder.df_finale.filter((F.col("lat") < -90) | (F.col("lat") > 90) | (F.col("lng") < -180) | (F.col("lng") > 180)).show()

    #Query
    query_manager = QueryManager(spark_builder)
    #query_manager.words_score_analysis(n=20)
    #query_manager.nationality_review_analysis(n=20)
    #query_manager.tag_influence_analysis(n=20)
    #query_manager.reviewer_behavior_analysis()
    #query_manager.outlier_analysis()
    #query_manager.review_length_analysis(n=50)
    
    #----------------------------------------------------------------ANALISI DELLA COERENZA----------------------------------------------------------------
    #spark_builder.df_finale.groupBy("Reviewer_Score").count().orderBy("Reviewer_Score").show()
    '''export_path="/Users/vincenzopresta/Desktop/Big Data/progetto/coherence_analysis"
    if not os.path.exists(export_path):
        os.makedirs(export_path)
        print(f"Cartella di esportazione creata: {export_path}")
    #query_manager.coherence_analysis(threshold=3.0, n=50, export_path=export_path)
    
    #DEBUG
    spark = SparkSession.builder \
    .appName("Visualizza CSV") \
    .getOrCreate()
    
    # Percorso al file CSV
    csv_path = "/Users/vincenzopresta/Desktop/Big Data/progetto/coherence_analysis/inconsistent_reviews_20250126_212714/part-00000-45f18a8d-c117-48fb-9a9f-0c91f9c300a6-c000.csv"
    # Carica il CSV come DataFrame
    df = spark.read.csv(csv_path, header=True, inferSchema=True)
    # Mostra le prime 20 righe del DataFrame
    df.select("Positive_Review","Positive_Review_Clean","Negative_Review","Negative_Review_Clean","Reviewer_Score","adjusted_prediction","error").show(20, truncate=True)

    df_original = spark_builder.df_finale.select("Positive_Review","Negative_Review","Reviewer_Score")\
                                                .where((col("Positive_Review").contains("No Positive")) & (col("Negative_Review").contains("The bathrooms can")))\
                                                .show(truncate=False)'''
                                                
    #----------------------------------------------------------------
    #query_manager.recovery_time_analysis(n=100)
    #query_manager.reputation_analysis()
    #query_manager.seasonal_sentiment_analysis(n=100)
    #query_manager.anomaly_detection(100)
    #query_manager.location_influence(10)