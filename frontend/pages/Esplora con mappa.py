
from backend import SparkBuilder
import streamlit as st
import folium
from streamlit_folium import st_folium
from pyspark.sql.functions import col
import pandas as pd
import plotly.express as px

import io

st.set_page_config(page_title="Hotel Map", layout="wide")

@st.cache_resource
def getSpark(appName):
    return SparkBuilder("appName")

spark = getSpark("BigData_App")

dataframe = spark.df_finale

df_temp = dataframe.select("hotel_name", "hotel_address", "lat", "lng").distinct().cache()

df = df_temp.toPandas()

@st.cache_data
def createMap(city):
    mappa = folium.Map(location=city_coords[city], zoom_start = 12)
    for _, row in df.iterrows():
        folium.Marker(
            location=[row['lat'], row['lng']],
            popup=f"<b>{row['hotel_name']}</b><br>Latitudine: {row['lat']}<br>Longitudine: {row['lng']}<br><br>Indirizzo:{row['hotel_address']}<br>",
            tooltip=row['hotel_name'],
            icon=folium.Icon(color='darkblue', icon='fa-solid fa-hotel', prefix='fa'),
        ).add_to(mappa)
    return mappa

@st.cache_data
def trend_singolo(hotel):
    # Supponiamo che spark.queryManager.trend_mensile() restituisca una figura Matplotlib
    fig = spark.queryManager.trend_mensile(hotel)
    
    # Salva la figura in un oggetto BytesIO
    img_stream = io.BytesIO()
    fig.savefig(img_stream, format='png')
    img_stream.seek(0)  # Torna all'inizio dello stream
    
    return img_stream

@st.cache_data
def anomalie(hotel):
    extreme_reviews = spark.queryManager.anomaly_detection(hotel)
    return extreme_reviews.toPandas()

@st.cache_data
def reputation(hotel):
    reputation_single = spark.queryManager.reputation_analysis_single(hotel)
    return reputation_single.toPandas()

@st.cache_data
def nearby_hotel_compare(lat, lng, d=1000):
    # Recupera i dati degli hotel vicini
    nearby_hotels = spark.queryManager.get_nearby_hotels(lat, lng, d)
    
    # Genera la figura con il grafico
    fig = spark.queryManager.trend_mensile_compare(nearby_hotels)
    
    # Salva la figura in un buffer BytesIO
    img_stream = io.BytesIO()
    fig.savefig(img_stream, format='png')
    img_stream.seek(0)  # Torna all'inizio dello stream
    
    return img_stream

#Sentiment analysis
@st.cache_data
def analyze_hotel_sentiment(hotel):
    sentiment = spark.queryManager.analyze_hotel_sentiment(hotel)
    return sentiment


if "stats" not in st.session_state:
    st.session_state.stats = spark.queryManager.hotelStatistics()

@st.cache_data
def getStats(hotel):
    res = st.session_state.stats.filter(col("Hotel_Name") == hotel)
    return res.toPandas()

# Configurare la pagina di Streamlit
st.title("Mappa degli Hotel")

#Coordinate città
city_coords = {
    "Milano": [45.4642, 9.1900],
    "Vienna": [48.2082, 16.3738],
    "Barcellona": [41.3851, 2.1734],
    "Londra": [51.5074, -0.1278],
    "Parigi": [48.8566, 2.3522],
    "Amsterdam" : [52.3709, 4.8902]
}

sentiment_emoji = {
    "Piuttosto Positivo": "📈",
    "Piuttosto Negativo": "📉",
    "Molto Positivo": "🤩",
    "Molto Negativo": "😓",
    "Neutrale": "😐"
    }

st.sidebar.title("Seleziona una città")
selected_city = st.sidebar.radio("Città", list(city_coords.keys()))

st.sidebar.title("🔍 Navigazione")
st.sidebar.markdown("### Sezioni disponibili:")

st.sidebar.markdown("- 🏠 **Home**")
st.sidebar.markdown("- 🗺️ **Esplora con mappa**")
st.sidebar.markdown("- 📍**Esplora per punto di interesse**")
st.sidebar.markdown("- #️⃣ **Esplora per tag**")
st.sidebar.markdown("- 🇮🇹 **Recensione-Nazionalità**")
st.sidebar.markdown("- 🏖️ **Sentiment Stagionale**")

# Creare una mappa con Folium
mappa = folium.Map(location=city_coords[selected_city], zoom_start=12)


# Mostrare la mappa in Streamlit
map_data = st_folium(createMap(selected_city), width=1000, height=550)

# Controllo se l'utente ha selezionato un marker
if map_data and map_data.get('last_object_clicked_tooltip') != None:
    hotel_selezionato = map_data.get('last_object_clicked_tooltip')
    st.markdown(
        ''' 
        # Informazioni 
        
        '''
    )
    
    print(hotel_selezionato)
    
    sentiment_result = analyze_hotel_sentiment(hotel_selezionato)
    st.metric(label="Sentiment Complessivo", label_visibility="visible", value=sentiment_result, delta=sentiment_emoji.get(sentiment_result, ""))
        
    with st.spinner(f" Individuo le statistiche per {hotel_selezionato}..."):
        st.write(f"*Statistiche per {hotel_selezionato}:*")
        stats = getStats(hotel_selezionato)
        #st.table(stats)
        # Creiamo due colonne per affiancare le metriche
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            st.metric(label="Total Reviews", value=stats.loc[0, "Total_Reviews"])
            st.metric(label="Total Positive Reviews", value=stats.loc[0, "Total_Positive_Reviews"])
            st.metric(label="Total Negative Reviews", value=stats.loc[0, "Total_Negative_Reviews"])
        
        with col2:
            # Creiamo il DataFrame per il grafico
            bar_data = pd.DataFrame({
            "Type": ["Positive Reviews", "Negative Reviews"],
            "Count": [stats["Total_Positive_Reviews"].sum(), stats["Total_Negative_Reviews"].sum()]
            })

            # Grafico a barre con Plotly
            fig = px.bar(
                bar_data,
                x="Type",
                y="Count",
                color="Type",
                text="Count",
                title="Numero di Recensioni Positive e Negative",
                color_discrete_map={"Positive Reviews": "green", "Negative Reviews": "red"},
            )

            # Mostriamo il grafico
            st.plotly_chart(fig, use_container_width=True)

            # Seconda colonna
        with col3:
            st.metric(label="Average Score", value=round(stats.loc[0, "Avg_Reviewer_Score"], 2))
            st.metric(label="Minimum reviewer score", value=stats.loc[0, "Min_Reviewer_Score"])
            st.metric(label="Maximum Reviewer score", value=stats.loc[0, "Max_Reviewer_Score"])


    with st.spinner(f"Generazione del trend mensile per {hotel_selezionato}..."):
        st.write(f"**Trend Mensile per {hotel_selezionato}:**")
        plt = trend_singolo(hotel_selezionato)
        st.image(plt)

    with st.spinner(f"Individuazione delle recensioni anomale per {hotel_selezionato}..."):
        extreme_reviews = anomalie(hotel_selezionato) 
        st.write(f"**Recensioni anomale rispetto alla media per {hotel_selezionato}:**")
            
        if not extreme_reviews.empty: 
            st.dataframe(extreme_reviews)    
        else: 
            st.write(f"Nessuna recensione anomala rispetto alla media per {hotel_selezionato}.")


    with st.spinner(f"Confronto ultime recensioni con la media per {hotel_selezionato}..."):
        st.write(f"**Confronto ultime recensioni con la media per {hotel_selezionato}:**")
        df_reputation = reputation(hotel_selezionato)
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Average Score", value=round(df_reputation.loc[0, "avg_historical_score"], 5))
            st.metric(label="Average Score recente", value=round(df_reputation.loc[0, "avg_recent_score"], 5))
        with col2:
            diff = round(df_reputation.loc[0, "score_difference"], 5)
            trend = "📈 L'hotel sta migliorando" if diff > 0 else "📉 L'hotel sta peggiorando"
            st.metric(label="Differenza", value=diff)
            st.write(trend)

    with st.spinner(f"Confronto con hotel vicini (nel raggio di 1km) a {hotel_selezionato}..."):
        hotel_lat = map_data.get("last_object_clicked").get("lat")
        hotel_lng = map_data.get("last_object_clicked").get("lng")
        st.write(f"**Trend hotel vicini (distanza 1km) a {hotel_selezionato}:**")
        nearby_hotel = spark.queryManager.get_nearby_hotels(hotel_lat, hotel_lng, 1000)
        nearby_hotel = nearby_hotel.select("Hotel_Name").distinct()
        if nearby_hotel.count()>1:
            st.image(nearby_hotel_compare(hotel_lat, hotel_lng, 1000))
        else:
            st.write(f"Nessun hotel vicino (distanza 1km) a {hotel_selezionato} trovato.")