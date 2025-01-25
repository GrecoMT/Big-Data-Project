from spark_builder import SparkBuilder
from query_manager import QueryManager
from pyspark.sql import functions as F

if __name__ == '__main__':

    dataset_path = "dataset/Hotel_Reviews.csv"
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
    
