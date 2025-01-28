import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import streamlit as st
from query_manager import QueryManager
from spark_builder import SparkBuilder
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from streamlit_folium import st_folium

#dataset_path = "/Users/vincenzopresta/Desktop/Big Data/dataset/Hotel_Reviews.csv"
dataset_path = "/Users/matteog/Documents/Università/Laurea Magistrale/Big Data/Progetto/Dataset/Hotel_Reviews.csv"

st.set_page_config(
    page_title="Hotel Dataset Analysis",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.sidebar.success("Select a demo above.")

@st.cache_resource
def get_spark_and_query_manager():
    """Inizializza SparkBuilder e QueryManager."""
    spark_builder = SparkBuilder("BigDataProject", dataset_path)
    #query_manager = QueryManager(spark_builder)
    query_manager = spark_builder.queryManager
    return query_manager

def clear_terminal():
    os.system('cls' if os.name == 'nt' else 'clear')
    
def show_landing_page():
    st.title("Benvenuti a radio 24! oggi abbiamo il brasiliano!")
    st.markdown(
        """
        ### Scopo del Progetto
        Questo progetto si concentra sull'analisi delle recensioni degli hotel per ottenere 
        informazioni significative e identificare tendenze, anomalie e comportamenti interessanti 
        dei recensori. Utilizzando tecnologie di Big Data come **Apache Spark**, combinate con 
        strumenti di visualizzazione come **Streamlit**, possiamo:
        - Esplorare dati geografici e clustering degli hotel.
        - Individuare recensioni sospette o anomalie nei punteggi.
        - Analizzare la correlazione tra nazionalità dei recensori e punteggi dati.
        - Esaminare l'influenza di tag e parole chiave sulle recensioni.
        - Studiare la reputazione e la percezione degli hotel nel tempo.

        ### Dati Utilizzati
        Il dataset utilizzato contiene:
        - **Recensioni di hotel** in Europa.
        - Attributi come punteggi, nazionalità dei recensori, contenuto delle recensioni, 
          data della recensione e informazioni geografiche (latitudine e longitudine).
        - Oltre **500.000 recensioni**, fornendo un'ampia base per approfondire diverse analisi.

        ### Obiettivi Principali
        - Rivelare tendenze nei punteggi e nel comportamento dei recensori.
        - Migliorare la comprensione dei dati delle recensioni per supportare decisioni 
          strategiche nel settore alberghiero.
        - Visualizzare i risultati in modo chiaro e interattivo.

        ---
        """,
        unsafe_allow_html=True
    )

    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/4/45/Hilton_Paris_Opera_Hotel_Entrance.jpg", 
        caption="Analisi delle recensioni di hotel in Europa",
        use_column_width=True
    )

    # Call-to-action
    st.markdown(
        """
        ### Come Utilizzare l'App
        Usa la barra laterale per navigare tra le diverse pagine:
        - **Geographic Locations**: Esplora la distribuzione geografica degli hotel.
        - **Anomaly Detection**: Identifica recensioni sospette.
        - **Word Cloud**: Analizza le parole più frequenti nelle recensioni.
        - **Tag Influence**: Scopri l'influenza delle tag sui punteggi degli hotel.
        - **Reputation Analysis**: Studia la reputazione degli hotel nel tempo.
        - ... e molto altro!

        ---
        **Inizia ora selezionando una pagina dal menu laterale!**
        """
    )    

# Main app con pulsanti
def main():
    
    clear_terminal()
    
    show_landing_page()
    
    st.title("Hotel Dataset Analysis")

if __name__ == "__main__":
    main()