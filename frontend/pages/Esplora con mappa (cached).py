from backend import SparkBuilder
import streamlit as st
import folium
from streamlit_folium import st_folium

import utils

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
    # Aggiungere marker per ogni hotel
    for _, row in df.iterrows():
        folium.Marker(
            location=[row['lat'], row['lng']],
            popup=f"<b>{row['hotel_name']}</b><br>Latitudine: {row['lat']}<br>Longitudine: {row['lng']}<br><br>Indirizzo:{row['hotel_address']}<br>",
            tooltip=row['hotel_name'],
            icon=folium.Icon(color='blue', icon='info-sign'),
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

st.sidebar.title("Seleziona una città")
selected_city = st.sidebar.radio("Città", list(city_coords.keys()))

st.sidebar.title("🔍 Navigazione")
st.sidebar.markdown("### Sezioni disponibili:")

st.sidebar.markdown("- 🏠 **Home**")
st.sidebar.markdown("- 📍 **Mappa Hotel**")
st.sidebar.markdown("- 📊 **Trend & Analisi**")
st.sidebar.markdown("- 🔍 **Anomaly Detection**")
st.sidebar.markdown("- 📝 **Word Cloud**")

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
    
    with st.spinner(f"Generazione del trend mensile per {hotel_selezionato}..."):
        st.write(f"**Trend Mensile per {hotel_selezionato}:**")
        plt = trend_singolo(hotel_selezionato)
        st.image(plt)
    
    with st.spinner(f"Individuazione delle recensioni anomale per {hotel_selezionato}..."):
        extreme_reviews = anomalie(hotel_selezionato)
        st.write(f"**Recensioni anomale rispetto alla media per {hotel_selezionato}:**")
        st.dataframe(extreme_reviews)    
    
    with st.spinner(f"Confronto ultime recensioni con la media per {hotel_selezionato}..."):
        st.write(f"**Confronto ultime recensioni con la media per {hotel_selezionato}:**")
        st.table(reputation(hotel_selezionato))

    with st.spinner(f"Confronto con hotel vicini (nel raggio di 1km) a {hotel_selezionato}..."):
        hotel_lat = map_data.get("last_object_clicked").get("lat")
        hotel_lng = map_data.get("last_object_clicked").get("lng")
        st.write(f"**Trend hotel vicini (distanza 1km) a {hotel_selezionato}:**")
        st.image(nearby_hotel_compare(hotel_lat, hotel_lng, 1000))