from spark_builder import SparkBuilder
from query_manager import QueryManager
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
    export_path="/Users/vincenzopresta/Desktop/Big Data/progetto/coherence_analysis"
    if not os.path.exists(export_path):
        os.makedirs(export_path)
        print(f"Cartella di esportazione creata: {export_path}")
    query_manager.coherence_analysis(threshold=2.0, n=50, export_path=export_path)

    
