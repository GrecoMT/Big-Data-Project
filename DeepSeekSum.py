from pyspark.sql.functions import col

import ollama

class SummaryLLM:
    def __init__(self, dataframe):
        self.df = dataframe
    
    def getReviews(self, hotel_name):
        reviews_df = self.df.filter((col("Hotel_Name") == hotel_name) & 
                (col("Positive_Review") != "No Positive") & 
                (col("Negative_Review") != "No Negative")).orderBy(col("Total_Number_of_Reviews_Reviewer_Has_Given")).select("Negative_Review", "Positive_Review").limit(20) 
        
        reviews_Pandas = reviews_df.toPandas()
        reviews = ""
        for _, row in reviews_Pandas.iterrows():
            reviews += f"{row['Positive_Review']} ; {row['Negative_Review']}. "
        return reviews
    
    def getSummary(self, hotel_name):
        reviews = self.getReviews(hotel_name)
        response = ollama.chat(model='deepseek-r1:1.5b', messages=[
        {
            'role': 'user',
            'content':f'Give me a detailed and elegant summary of the following text:{reviews}',
        },
        ])
        summary = response['message']['content']
        thinking = "<think>"+summary.split('<think>')[1].split('</think>')[0]+"</think>"
        return summary.replace(thinking, "")

