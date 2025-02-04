import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader
from pyspark.sql import DataFrame
from sklearn.model_selection import train_test_split
from pyspark.sql.functions import when, col, abs, lit


class BertTrainer:

    def __init__(self, df: DataFrame, model_path="models/bert_model"):
        self.df = df
        self.model_path = model_path
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=1)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # usa gpu se c'è
        self.model.to(self.device)


    def preprocess_data(self):
        """ Caricamento dataset Spark e conversione in Pandas + Pulizia testo """
        df = self.df.select("Positive_Review", "Negative_Review", "Reviewer_Score").toPandas()

        df["Positive_Review"] = df["Positive_Review"].str.lower().str.replace(r"[^a-z\s]", "", regex=True)
        df["Negative_Review"] = df["Negative_Review"].str.lower().str.replace(r"[^a-z\s]", "", regex=True)
        df["Review_Text"] = df["Positive_Review"] + " " + df["Negative_Review"]

        return df
    
    def infer_review_score(self, review_text):
        """ Usa BERT per predire il punteggio di una recensione """
        encoding = {key: val.to(self.device) for key, val in encoding.items()} #spostato su gpu
        with torch.no_grad():
            output = self.model(**encoding)
            score = output.logits.squeeze().item()
        return max(0, min(10, round(score, 1)))  # Limita tra 0 e 10
    
    def analyze_consistency(self, threshold=2.0, n=10, export_path=None):
        """
        Analizza la coerenza tra il punteggio reale e quello predetto dal modello BERT.
        """
        df_clean = self.preprocess_data()
        df_clean["Predicted_Score"] = df_clean["Review_Text"].apply(self.infer_review_score)

        # Converte il DataFrame Pandas in Spark DataFrame
        df_spark = self.df.withColumn("Predicted_Score", lit(None).cast("double"))
        for idx, row in df_clean.iterrows():
            df_spark = df_spark.withColumn(
                "Predicted_Score",
                when(col("Positive_Review") == row["Positive_Review"], lit(row["Predicted_Score"])).otherwise(col("Predicted_Score"))
            )

        # Calcola errore assoluto tra punteggio predetto e reale
        df_spark = df_spark.withColumn("error", abs(col("Predicted_Score") - col("Reviewer_Score")))

        # Filtra recensioni incoerenti (dove errore > threshold)
        inconsistent_reviews = df_spark.filter(col("error") > threshold)

        # Rimuove colonne non necessarie
        columns_to_drop = ["Predicted_Score"]
        for col_name in columns_to_drop:
            if col_name in inconsistent_reviews.columns:
                inconsistent_reviews = inconsistent_reviews.drop(col_name)

        # Mostra le recensioni incoerenti
        print(f"\nRecensioni incoerenti (errore > {threshold}):")
        inconsistent_reviews.select("Positive_Review", "Negative_Review", "Reviewer_Score", "error").show(n, truncate=True)

        # Esporta se richiesto
        if export_path:
            inconsistent_reviews.write.mode("overwrite").csv(export_path, header=True)
            print(f"\nRecensioni incoerenti esportate in: {export_path}")

        return inconsistent_reviews

    class HotelReviewDataset(Dataset):
        def __init__(self, reviews, scores, tokenizer, max_len=256):
            self.reviews = reviews
            self.scores = scores
            self.tokenizer = tokenizer
            self.max_len = max_len

        def __len__(self):
            return len(self.reviews)

        def __getitem__(self, idx):
            text = self.reviews[idx]
            score = self.scores[idx]

            encoding = self.tokenizer(
                text, 
                padding="max_length", 
                truncation=True, 
                max_length=self.max_len, 
                return_tensors="pt"
            )

            return {
                "input_ids": encoding["input_ids"].squeeze(),
                "attention_mask": encoding["attention_mask"].squeeze(),
                "labels": torch.tensor(score, dtype=torch.float)
            }

    def train_model(self):
        """ Addestramento BERT per regressione """

        df_clean = self.preprocess_data()

        # Splitting dati
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            df_clean["Review_Text"], df_clean["Reviewer_Score"], test_size=0.2, random_state=42
        )

        train_dataset = self.HotelReviewDataset(train_texts.tolist(), train_labels.tolist(), self.tokenizer)
        val_dataset = self.HotelReviewDataset(val_texts.tolist(), val_labels.tolist(), self.tokenizer)

        training_args = TrainingArguments(
            output_dir=self.model_path,
            num_train_epochs=3,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            fp16=True,                          #imposta questo a false se addestri su cpu (non succederà mai)
            gradient_accumulation_steps=4,
            save_total_limit=1,
            warmup_steps=300,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=10,
            evaluation_strategy="epoch"
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset
        )
        print("Device:", self.device)
        trainer.train()

        # Salvataggio modello
        self.model.save_pretrained(self.model_path)
        self.tokenizer.save_pretrained(self.model_path)

        print("Modello BERT addestrato e salvato!")

