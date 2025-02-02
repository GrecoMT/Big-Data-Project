import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import streamlit as st

from backend import SparkBuilder

# Configurazione avanzata della pagina
st.set_page_config(
    page_title="Hotel Dataset Analysis",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.sidebar.title("🔍 Navigazione")
st.sidebar.markdown("### Sezioni disponibili:")

st.sidebar.markdown("- 🏠 **Home**")
st.sidebar.markdown("- 📍 **Mappa Hotel**")
st.sidebar.markdown("- 📊 **Trend & Analisi**")
st.sidebar.markdown("- 🔍 **Anomaly Detection**")
st.sidebar.markdown("- 📝 **Word Cloud**")

@st.cache_resource
def get_spark_and_query_manager():
    """Inizializza SparkBuilder e QueryManager."""
    spark_builder = SparkBuilder("BigDataProject")
    query_manager = spark_builder.queryManager
    return query_manager

def clear_terminal():
    os.system('cls' if os.name == 'nt' else 'clear')

# Funzione per la homepage migliorata
def show_landing_page():
    
    st.markdown(
        "<h1 style='text-align: center; color: #4A90E2;'>🔍 Analisi delle Recensioni degli Hotel</h1>", 
        unsafe_allow_html=True
    )
    
    col1, col2, col3 = st.columns([1,2,1])
    
    with col2:
        st.image("frontend/images/PNY_Exterior_with_Rolls_Royce.jpg", 
                 width=700, caption="Visualizzazione dei dati sulle recensioni degli hotel")
    
  
    # Layout con colonne per una presentazione più moderna
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(
            """
            <h3 style='color: #333;'>📌 Scopo del Progetto</h3>
            <p style='font-size: 18px;'>
            Questo progetto sfrutta Spark per analizzare le recensioni degli hotel in Europa, individuare 
            anomalie e tendenze e fornire insight significativi per il settore alberghiero.
            </p>
            """, unsafe_allow_html=True
        )

        st.markdown("### 🚀 Tecnologie Utilizzate")
        st.markdown("- 🔥 **Apache Spark** per il processamento massivo dei dati")
        st.markdown("- 🎨 **Streamlit** per un’interfaccia intuitiva e interattiva")
        st.markdown("- 📌 **Folium & Matplotlib** per la visualizzazione dei dati")

    # Linea divisoria
    st.markdown("---")

    # Obiettivi principali
    st.markdown(
        """
        <h3 style='color: #333;'>🎯 Obiettivi Principali</h3>
        """, unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("✅ **Esplorazione geografica** degli hotel")
        st.markdown("✅ **Identificazione di recensioni sospette**")
    with col2:
        st.markdown("✅ **Analisi delle parole chiave** nelle recensioni")
        st.markdown("✅ **Andamento della reputazione degli hotel** nel tempo")
    with col3:
        st.markdown("✅ **Visualizzazione interattiva** dei dati")
        st.markdown("✅ **Correlazione tra nazionalità e punteggi**")

    # Divider
    st.markdown("---")

    # Sezione interattiva con pulsanti
    st.markdown("<h3 style='text-align: center;'>💡 Inizia ora selezionando una sezione dal menu laterale!</h3>", unsafe_allow_html=True)

# Gestione delle sezioni
def main():
    clear_terminal()
    
    show_landing_page()
    
if __name__ == "__main__":
    main()



