from backend import SparkBuilder
from pyspark.sql.functions import col, lower
import os 

def clear_terminal():
    os.system('cls' if os.name == 'nt' else 'clear')


if __name__ == '__main__':
    
    clear_terminal()
    
    #export_path="/Users/vincenzopresta/Desktop/Big Data/progetto/coherence_analysis"

    spark_builder = SparkBuilder(appname="BigData_App")

    # Controlla se i dati sono puliti
    #spark_builder.df_finale.select([F.count(F.when(F.col(c).isNull(), 1)).alias(c) for c in spark_builder.df_finale.columns]).show()
    #spark_builder.df_finale.filter((F.col("lat") < -90) | (F.col("lat") > 90) | (F.col("lng") < -180) | (F.col("lng") > 180)).show()

    #Query
    #query_manager = QueryManager(spark_builder)
    
    query_manager = spark_builder.queryManager

    #query_manager.words_score_analysis(n=20)
    #query_manager.nationality_review_analysis(n=20)
    #query_manager.tag_influence_analysis(n=20)
    #query_manager.outlier_analysis()
    #query_manager.review_length_analysis(n=50)
    #query_manager.recovery_time_analysis(n=100)
    #query_manager.reputation_analysis(n=100)
    #query_manager.seasonal_sentiment_analysis(n=100)
    #x = query_manager.nearby_hotels(45.48244094848633,9.175698280334473)
    #x.show(20, truncate=True)
    #query_manager.location_influence(10)
    '''
    result = query_manager.get_nearby_hotels(45.464098, 9.191926)
    dff = result.toPandas()
    print(dff.head())'''
    query_manager.tag_influence_analysis()