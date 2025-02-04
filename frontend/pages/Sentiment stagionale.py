import streamlit as st
import pandas as pd
import plotly.express as px
from pyspark.sql.functions import avg, count, col
from season_sentiment_analysis import SeasonSentimentAnalysis

# Funzione per caricare i dati della sentiment analysis stagionale


@st.cache_resource
def getSpark(appName):
    from backend import SparkBuilder
    return SparkBuilder(appName)

spark = getSpark("BigData_App")

st.sidebar.title("🔍 Navigazione")
st.sidebar.markdown("### Sezioni disponibili:")

st.sidebar.markdown("- 🏠 **Home**")
st.sidebar.markdown("- 📍 **Mappa Hotel**")
st.sidebar.markdown("- 📊 **Trend & Analisi**")
st.sidebar.markdown("- 🔍 **Anomaly Detection**")
st.sidebar.markdown("- 📝 **Word Cloud**")

def load_seasonal_sentiment():
    sentiment_analysis = SeasonSentimentAnalysis(spark.df_finale)
    df_preprocessed = sentiment_analysis.preprocess()
    
    seasonal_sentiment = df_preprocessed.groupBy("Hotel_Name", "Hotel_Address", "Season").agg(
        avg("Net_Sentiment").alias("avg_sentiment"),
        avg("Reviewer_Score").alias("avg_reviewer_score"),
        count("*").alias("review_count")
    ).orderBy("Hotel_Name", "Season")

    return seasonal_sentiment

# 🔹 Layout di Streamlit
st.set_page_config(page_title="Analisi Sentiment Stagionale", layout="wide")

# 🔹 Carica i dati
df_seasonal = load_seasonal_sentiment()

# 🔹 Titolo e descrizione
st.title("📊 Analisi Sentiment Stagionale negli Hotel")
st.write(
    "Questa dashboard mostra come il **sentiment medio** delle recensioni varia **in base alle stagioni** "
    "e confronta il **punteggio dato dagli utenti**. Seleziona una città e successivamente un hotel per vedere le variazioni!"
)

if "selected_button" not in st.session_state:
    st.session_state.selected_button = None

col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    button1 = st.button('Milano')
with col2:
    button2 = st.button('Vienna')
with col3:
    button3 = st.button('Barcellona')
with col4:
    button4 = st.button('Londra')
with col5:
    button5 = st.button('Parigi')
with col6:
    button6 = st.button('Amsterdam')
    
if button1:
    st.session_state.selected_button = "Milan"
elif button2:
    st.session_state.selected_button = "Vienna"
elif button3:
    st.session_state.selected_button = "Barcelona"
elif button4:
    st.session_state.selected_button = "London"
elif button5:
    st.session_state.selected_button = "Paris"
elif button6:
    st.session_state.selected_button = "Amsterdam"
    
city = st.session_state.selected_button 
    
hotels_city = None
if st.session_state.selected_button != None: 
    hotels_city = df_seasonal.filter(col("Hotel_Address").contains(city))

    # 🔹 Seleziona un hotel specifico
    hotels_name = hotels_city.select(col("Hotel_Name")).distinct()
    
    hotel_selected = st.selectbox(f"🏨 Seleziona un hotel di {city}", options=hotels_name, placeholder="Seleziona un hotel", index=None)

    if hotel_selected != None:
        # 🔹 Filtra il dataset per l'hotel selezionato
        df_hotel = hotels_city.filter(col("Hotel_Name") == hotel_selected).toPandas()

        # 🔹 Grafico interattivo - Sentiment e Punteggio per Stagione
        fig = px.line(
            df_hotel,
            x="Season",
            y=["avg_sentiment", "avg_reviewer_score"],
            markers=True,
            labels={"value": "Punteggio", "Season": "Stagione"},
            title=f"📈 Sentiment e Punteggio per Stagione - {hotel_selected}",
        )

        # 🔹 Personalizzazione del grafico
        fig.update_layout(
            xaxis_title="Stagione",
            yaxis_title="Punteggio Medio",
            legend_title="Metrica",
            template="plotly_dark",
            hovermode="x unified",
        )

        # 🔹 Mostra il grafico
        st.plotly_chart(fig, use_container_width=True)

        # 🔹 Tabella dati aggregati
        st.write("📊 **Dati Aggregati delle Recensioni**")
        st.dataframe(df_hotel)

        # 🔹 Box con insights
        st.success(
            f"🔎 **Insight:** L'hotel **{hotel_selected}** ha un **sentiment medio di {df_hotel['avg_sentiment'].mean():.2f}** "
            f"e un **punteggio utenti di {df_hotel['avg_reviewer_score'].mean():.2f}** durante l'anno."
        )

        # 🔹 Footer
        st.markdown(
            "💡 Questa analisi aiuta a identificare **pattern stagionali** nella percezione degli hotel da parte degli utenti!"
        )
