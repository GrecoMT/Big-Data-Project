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
st.sidebar.markdown("- 📊 **Analisi delle parole**")
st.sidebar.markdown("- 🗺️ **Esplora con mappa**")
st.sidebar.markdown("- 📍 **Esplora per punto di interesse**")
st.sidebar.markdown("- #️⃣ **Esplora per tag**")
st.sidebar.markdown("- 🇮🇹 **Recensione-Nazionalità**")
st.sidebar.markdown("- 🏖️ **Sentiment Stagionale**")

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
    "<h1 style='text-align: center; color: #000000;'>🔍 Analisi delle Recensioni degli Hotel</h1>", 
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
            Questo progetto sfrutta Spark per analizzare le recensioni degli hotel in Europa.
            </p>
            """, unsafe_allow_html=True
        )

        st.markdown("### 🚀 Tecnologie Utilizzate")
        st.markdown("- 🔥 **Apache Spark** per il processamento massivo dei dati")
        st.markdown("- 🎨 **Streamlit** per un’interfaccia intuitiva e interattiva")
        st.markdown("- 📌 **Folium & Matplotlib** per la visualizzazione dei dati")
        st.markdown("- 🧠 **RoBERTa** per l'analisi del sentiment delle recensioni")
        st.markdown("- 🤖 **DeepSeek** per la generazione dei riassunti delle recensioni")

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
        st.markdown("✅ **Analisi del testo delle recensioni**")
        st.markdown("✅ **Valutazione della reputazione degli hotel**")
    with col2:
        st.markdown("✅ **Analisi del sentiment**")
        st.markdown("✅ **Individuazione di recensioni sospette**")
    with col3:
        st.markdown("✅ **Analisi dei trend temporali**")
        st.markdown("✅ **Correlazione tra nazionalità e punteggi**")

    # Divider
    st.markdown("---")

    # Sezione interattiva con pulsanti
    st.markdown("<h3 style='text-align: center;'>💡 Inizia ora selezionando una pagina dal menu laterale!</h3>", unsafe_allow_html=True)

# Gestione delle sezioni
def main():
    clear_terminal()
    
    show_landing_page()
    
if __name__ == "__main__":
    main()



