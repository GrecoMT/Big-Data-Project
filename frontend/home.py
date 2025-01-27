import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import streamlit as st
from query_manager import QueryManager
from spark_builder import SparkBuilder
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from streamlit_folium import st_folium

st.set_page_config(
    page_title="Hotel Dataset Analysis",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def get_spark_and_query_manager():
    """Inizializza SparkBuilder e QueryManager."""
    spark_builder = SparkBuilder("BigDataProject", "/Users/vincenzopresta/Desktop/Big Data/dataset/Hotel_Reviews.csv")
    query_manager = QueryManager(spark_builder)
    return query_manager

def clear_terminal():
    os.system('cls' if os.name == 'nt' else 'clear')

# Funzione per ogni sezione/analisi
def show_geographic_locations(query_manager):
    st.subheader("Geographic Locations")
    st.markdown("La mappa mostra i cluster geografici degli hotel.")
    map_ = query_manager.location_influence()
    st_folium(map_, width=1000)

def show_anomaly_detection(query_manager):
    st.subheader("Anomaly Detection")
    st.markdown("Identificazione di recensioni sospette.")
    anomalies = query_manager.anomaly_detection(n=20)
    st.dataframe(anomalies)

def show_word_cloud(query_manager):
    st.subheader("Word Cloud Analysis")
    st.markdown("Analisi delle frequenze delle parole nelle recensioni.")
    dswords = query_manager.words_score_analysis(n=20)  # Chiamata alla query
    word_frequencies = dswords.toPandas().set_index("word")["avg_score"].to_dict()  # Frequenze delle parole
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_frequencies)
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)

def show_nationality_analysis(query_manager):
    st.subheader("Nationality Review Analysis")
    st.markdown("Analisi della correlazione tra nazionalità e recensioni.")
    nationality_data = query_manager.nationality_review_analysis(n=20)
    st.dataframe(nationality_data)

def show_tag_influence(query_manager):
    st.subheader("Tag Influence Analysis")
    st.markdown("Influenza delle tag sui punteggi degli hotel.")
    tag_data = query_manager.tag_influence_analysis(n=20)
    st.dataframe(tag_data)

def show_reputation_analysis(query_manager):
    st.subheader("Reputation Analysis")
    st.markdown("Analisi della reputazione degli hotel nel tempo.")
    reputation_data = query_manager.reputation_analysis()
    st.dataframe(reputation_data)

def show_recovery_time_analysis(query_manager):
    st.subheader("Recovery Time Analysis")
    st.markdown("Analisi del tempo necessario per recuperare da una percezione negativa.")
    recovery_data = query_manager.recovery_time_analysis(n=20)
    st.dataframe(recovery_data)

# Main app con pulsanti
def main():
    
    clear_terminal()
    
    st.title("Hotel Dataset Analysis")
    query_manager = get_spark_and_query_manager()

    st.sidebar.title("Sezioni")
    option = st.sidebar.radio(
        "Seleziona un'analisi da eseguire:",
        (
            "Geographic Locations",
            "Anomaly Detection",
            "Word Cloud Analysis",
            "Nationality Analysis",
            "Tag Influence",
            "Reputation Analysis",
            "Recovery Time Analysis"
        )
    )

    # Mostra la sezione selezionata
    if option == "Geographic Locations":
        show_geographic_locations(query_manager)
    elif option == "Anomaly Detection":
        show_anomaly_detection(query_manager)
    elif option == "Word Cloud Analysis":
        show_word_cloud(query_manager)
    elif option == "Nationality Analysis":
        show_nationality_analysis(query_manager)
    elif option == "Tag Influence":
        show_tag_influence(query_manager)
    elif option == "Reputation Analysis":
        show_reputation_analysis(query_manager)
    elif option == "Recovery Time Analysis":
        show_recovery_time_analysis(query_manager)


if __name__ == "__main__":
    main()