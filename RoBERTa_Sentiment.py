import re
import torch
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from pyspark.sql.functions import col
from transformers import AutoTokenizer, AutoModelForSequenceClassification

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")

class RoBERTa_Sentiment:
    def __init__(self):
        self.model_name = "cardiffnlp/twitter-roberta-base-sentiment"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.eval()

        self.lemmatizer = WordNetLemmatizer()
        self.stopwords = set(stopwords.words("english"))

    def preprocess_text(self, text):
        """ pulizia, tokenizzazione, stopwords, lemmatizzazione."""
        if not text or text.lower() in ["no negative", "no positive"]:
            return ""

        text = text.lower()
        text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # Rimuove URL
        text = re.sub(r"<.*?>", "", text)  # Rimuove tag HTML
        text = re.sub(r"[^\w\s]", "", text)  # Rimuove punteggiatura
        text = re.sub(r"\d+", "", text)  # Rimuove numeri

        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in self.stopwords]
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens]

        return " ".join(tokens)

    def analyze_review_sentiment(self, positive_text, negative_text):
        """Combina il sentiment della recensione positiva e negativa per restituire un'unica valutazione finale."""

        strong_words = {"all", "everything", "always"}  # Parole che indicano giudizi forti
        neutral_negative_words = {"nothing", "none", "no negative"}  # Nessun problema
        neutral_positive_words = {"no positive"}  # Nessun aspetto positivo

        positive_text = positive_text.strip().lower()
        negative_text = negative_text.strip().lower()

        if negative_text in neutral_negative_words:
            negative_text = ""

        if positive_text in neutral_positive_words:
            positive_text = ""

        contains_strong_positive = any(word in positive_text.split() for word in strong_words)
        contains_strong_negative = any(word in negative_text.split() for word in strong_words)

        if contains_strong_negative and not contains_strong_positive:
            return "negative"

        if contains_strong_positive and not contains_strong_negative:
            return "positive"

        # Se solo una parte è significativa, usa solo quella 
        if positive_text and not negative_text:
            return self._predict_sentiment(positive_text)
        if negative_text and not positive_text:
            return self._predict_sentiment(negative_text)

        # Se entrambe le parti contengono testo utile, calcola il sentiment combinato
        pos_sentiment = self._predict_sentiment(positive_text)
        neg_sentiment = self._predict_sentiment(negative_text)

        # Mapping del sentiment in valori numerici
        sentiment_map = {"negative": -1, "neutral": 0, "positive": 1}
        pos_score = sentiment_map[pos_sentiment]
        neg_score = sentiment_map[neg_sentiment]

        # Calcolo dei pesi in base alla lunghezza del testo
        pos_length = len(positive_text) if positive_text else 1
        neg_length = len(negative_text) if negative_text else 1
        total_length = pos_length + neg_length

        pos_weight = pos_length / total_length
        neg_weight = neg_length / total_length

        # Media ponderata del sentiment
        combined_score = (pos_score * pos_weight) + (neg_score * neg_weight)

        if combined_score > 0.3:
            return "positive"
        elif combined_score < -0.3:
            return "negative"
        else:
            return "neutral"


    def _predict_sentiment(self, text):
        clean_text = self.preprocess_text(text)
        if not text.strip():
            return "neutral"
        
        inputs = self.tokenizer(clean_text, return_tensors="pt", truncation=True, padding=True, max_length=512)

        with torch.no_grad():
            outputs = self.model(**inputs)

        predicted_class = torch.argmax(outputs.logits, dim=-1).item()
        labels = {0: "negative", 1: "neutral", 2: "positive"}

        return labels[predicted_class]

    def analyze_hotel_sentiment(self, hotel_name, df):
                
        hotel_reviews = df.filter(col("Hotel_Name") == hotel_name).select("Positive_Review", "Negative_Review").collect()

        if not hotel_reviews:
            return "Nessuna recensione"

        sentiment_scores = []
        sentiment_map = {"negative": -1, "neutral": 0, "positive": 1}

        for row in hotel_reviews:
            print(f"analizzando: {row}")
            positive_text = row["Positive_Review"].strip()
            negative_text = row["Negative_Review"].strip()

            # Calcola il sentiment della recensione utilizzando il metodo per la singola recensione
            review_sentiment = self.analyze_review_sentiment(positive_text, negative_text)
            
            print(f"sentiment = {review_sentiment}")

            # Mappa il sentiment in valori numerici e accumula
            sentiment_scores.append(sentiment_map[review_sentiment])

        avg_score = sum(sentiment_scores) / len(sentiment_scores)

        # Determina il sentiment complessivo basato sulla media
        print(avg_score)
        
        if avg_score > 0.2 and avg_score < 0.4:
            return "Piuttosto Positivo"
        elif avg_score >= 0.4:
            return "Molto Positivo"
        elif avg_score < -0.2 and avg_score > - 0.3:
            return "Piuttosto Negativo"
        elif avg_score <= -0.4:
            return "Molto Negativo"
        else:
            return "Neutrale"
