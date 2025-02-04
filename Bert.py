import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader
from pyspark.sql import DataFrame
from sklearn.model_selection import train_test_split
from pyspark.sql.functions import when, col, abs, lit
import os

class BertTrainer:

    def __init__(self, df, model_path=None):
        self.df = df
        self.model_path = model_path
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        # Controlla se il modello pre-addestrato esiste nella directory passataa
        if os.path.exists(self.model_path) and os.path.isfile(os.path.join(self.model_path, "model.safetensors")):
            print(f"Caricamento del modello da {self.model_path}...")
            self.model = BertForSequenceClassification.from_pretrained(self.model_path, num_labels=1)
        else:
            print("Modello non trovato! Inizializzazione di un nuovo modello BERT.")
            self.model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=1)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
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
        # Tokenizza il testo di input
        encoding = self.tokenizer(
            review_text,
            padding="max_length",
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )
    
        encoding = {key: val.to(self.device) for key, val in encoding.items()} #spostato su gpu
        with torch.no_grad():
            output = self.model(**encoding)
            score = output.logits.squeeze().item()
        return max(0, min(10, round(score, 1)))  # Limita tra 0 e 10
    
    def analyze_consistency(self, threshold=2.0, n=10, export_path="models/bert_test"):
        """
        Analizza la coerenza tra il punteggio reale e quello predetto dal modello BERT.
        """
        print("🔄 Inizio analisi della coerenza...")
        df_clean = self.preprocess_data()
        print("✅ Pre-elaborazione completata.")
        
        print("🔍 Calcolo punteggi predetti...")
        df_clean["Predicted_Score"] = df_clean["Review_Text"].apply(self.infer_review_score)
        print("✅ Calcolo completato.")
        
        print("🔍 Calcolo errore assoluto...")
        df_spark = self.df.withColumn("Predicted_Score", lit(None).cast("double"))
        for idx, row in df_clean.iterrows():
            df_spark = df_spark.withColumn(
                "Predicted_Score",
                when(col("Positive_Review") == row["Positive_Review"], lit(row["Predicted_Score"])).otherwise(col("Predicted_Score"))
            )

        # Calcola errore assoluto tra punteggio predetto e reale
        df_spark = df_spark.withColumn("error", abs(col("Predicted_Score") - col("Reviewer_Score")))
        print("✅ Errore calcolato.")
        
        # Filtra recensioni incoerenti (dove errore > threshold)
        inconsistent_reviews = df_spark.filter(col("error") > threshold)

        print(f"✅ Filtrate {inconsistent_reviews.count()} recensioni incoerenti (errore > {threshold}).")
        
        inconsistent_reviews.select("Positive_Review", "Negative_Review", "Reviewer_Score", "Predicted_Score", "error").show(n, truncate=True)

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
        print("🔄 Inizio addestramento...")
        df_clean = self.preprocess_data()
        print("✅ Dati preprocessati.")

        # Splitting dati
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            df_clean["Review_Text"], df_clean["Reviewer_Score"], test_size=0.2, random_state=42
        )
        
        print(f"📊 Numero di campioni di training: {len(train_texts)}")
        print(f"📊 Numero di campioni di validazione: {len(val_texts)}")

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
        print(f"⚙️ Inizio addestramento su dispositivo: {self.device}")
        trainer.train()
        print("✅ Addestramento completato!")

        # Salvataggio modello
        self.model.save_pretrained(self.model_path)
        self.tokenizer.save_pretrained(self.model_path)
        print(f"✅ Modello salvato in: {self.model_path}")

