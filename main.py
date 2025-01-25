from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import IntegerType, FloatType, StringType, Row
from pyspark.sql.functions import (regexp_replace, split, expr, col, to_date, regexp_extract, udf, count,
                                   avg, sum, when, first, concat, max, min, countDistinct, explode, lower, lit, year,
                                   month, abs)
from pyspark.sql import functions as F

import utils

datasetPath = "/Users/matteog/Documents/Università/Laurea Magistrale/Big Data/Progetto/Dataset/Hotel_Reviews.csv"


class SparkBuilder:

    def __init__(self, appname: str):
        self.spark = (SparkSession.builder.master("local[*]").
                      appName(appname).getOrCreate())
        self.dataset = self.spark.read.csv(datasetPath, header=True, inferSchema=True)
        self.castDataset()
        self.cleanDataset()
        self.df_finale = self.fillLatLng()

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
        #df = df.withColumn("Tags", regexp_replace(col("Tags"), "[\[\]']", ""))
        df = df.withColumn("Tags", regexp_replace(col("Tags"), r"[\[\]']", ""))
        df = df.withColumn("Tags", split(col("Tags"), ", "))
        df = df.withColumn("Tags", expr("transform(Tags, x -> trim(x))"))

        self.dataset = df.cache()
    
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
        self.dataset = df.filter(col("Reviewer_Nationality") != " ")
    

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
            F.coalesce(self.dataset["lat"], df_reqfill["lat_f"])
        ).withColumn(
            "lng", 
            F.coalesce(self.dataset["lng"], df_reqfill["lng_f"])
        )
        # rimuovere le colonne duplicate di df1 ("lat" e "lng" di df1)
        ret = df_merged.drop("lat_f", "lng_f")
        return ret

if __name__ == '__main__':
    spark = SparkBuilder("BigData_App")
    print(spark.df_finale.toPandas().info())
    