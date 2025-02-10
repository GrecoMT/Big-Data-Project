import streamlit as st
import plotly.express as px
from pyspark.sql.functions import col


st.set_page_config(page_title="Analisi Sentiment Stagionale", layout="wide")

@st.cache_resource
def getSpark(appName):
    from backend import SparkBuilder
    return SparkBuilder(appName)

spark = getSpark("BigData_App")

st.sidebar.title("🔍 Navigazione")
st.sidebar.markdown("### Sezioni disponibili:")

st.sidebar.markdown("- 🏠 **Home**")
st.sidebar.markdown("- 📊 **Analisi delle parole**")
st.sidebar.markdown("- 🗺️ **Esplora con mappa**")
st.sidebar.markdown("- 📍 **Esplora per punto di interesse**")
st.sidebar.markdown("- #️⃣ **Esplora per tag**")
st.sidebar.markdown("- 🇮🇹 **Recensione-Nazionalità**")
st.sidebar.markdown("- 🏖️ **Sentiment Stagionale**")

@st.cache_resource()
def load_seasonal_sentiment():
    return spark.queryManager.seasonal_sentiment_analysis()

st.title("📊 Analisi Sentiment Stagionale negli Hotel")
st.write(
    "Questa dashboard mostra come il **sentiment medio** delle recensioni varia **in base alle stagioni** "
    "e confronta il **punteggio dato dagli utenti**. Seleziona una città e successivamente un hotel per vedere le variazioni!"
)

df_seasonal = load_seasonal_sentiment()

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

    hotels_name = hotels_city.select(col("Hotel_Name")).distinct()
    
    hotel_selected = st.selectbox(f"🏨 Seleziona un hotel di {city}", options=hotels_name, placeholder="Seleziona un hotel", index=None)

    if hotel_selected != None:
        df_hotel = hotels_city.filter(col("Hotel_Name") == hotel_selected).toPandas()

        # Grafico con sentiment e punteggio per stagione
        fig = px.line(
            df_hotel,
            x="Season",
            y=["avg_sentiment", "avg_reviewer_score"],
            markers=True,
            labels={"value": "Punteggio", "Season": "Stagione"},
            title=f"📈 Sentiment e Punteggio per Stagione - {hotel_selected}",
        )

        fig.update_layout(
        xaxis_title="Stagione",
        yaxis_title="Punteggio Medio",
        legend_title="Metrica",
        template="plotly_dark",
        hovermode="x unified",
    )
        st.plotly_chart(fig, use_container_width=True)
        
        sentiment_mean_score = df_hotel['avg_sentiment'].mean()

        if sentiment_mean_score >= 0.4:
            sentiment_label = "😊 Positivo"
            sentiment_color = "green"
        elif sentiment_mean_score < 0:
            sentiment_label = "😠 Negativo"
            sentiment_color = "red"
        else:
            sentiment_label = "😐 Neutrale"
            sentiment_color = "gray"
            
        st.markdown(
            f"<h3 style='text-align: center; color: {sentiment_color};'>📝 Sentiment Complessivo delle recensioni: {sentiment_label}</h3>",
            unsafe_allow_html=True
        )

        st.write("📊 **Dati Aggregati delle Recensioni**")
        st.dataframe(df_hotel)

        st.success(
            f"🔎 **Insight:** L'hotel **{hotel_selected}** ha un **sentiment medio di {df_hotel['avg_sentiment'].mean():.2f}** "
            f"e un **punteggio utenti di {df_hotel['avg_reviewer_score'].mean():.2f}** durante l'anno."
        )