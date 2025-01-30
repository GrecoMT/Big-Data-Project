from backend import SparkBuilder
import streamlit as st
import folium
from streamlit_folium import st_folium

import utils

st.set_page_config(page_title="Hotel Map", layout="wide")

@st.cache_resource
def getSpark(appName):
    return SparkBuilder("appName")

spark = getSpark("BigData_App")

dataframe = spark.df_finale

df_temp = dataframe.select("hotel_name", "hotel_address", "lat", "lng").distinct().cache()

df = df_temp.toPandas()

# Configurare la pagina di Streamlit

st.title("Mappa degli Hotel")

#Coordinate città
city_coords = {
    "Milano": [45.4642, 9.1900],
    "Vienna": [48.2082, 16.3738],
    "Barcellona": [41.3851, 2.1734],
    "Londra": [51.5074, -0.1278],
    "Parigi": [48.8566, 2.3522]
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

# Aggiungere marker per ogni hotel
for _, row in df.iterrows():
    folium.Marker(
        location=[row['lat'], row['lng']],
        popup=f"<b>{row['hotel_name']}</b><br>Latitudine: {row['lat']}<br>Longitudine: {row['lng']}<br><br>Indirizzo:{row['hotel_address']}<br>",
        tooltip=row['hotel_name'],
        icon=folium.Icon(color='blue', icon='info-sign'),
    ).add_to(mappa)

# Mostrare la mappa in Streamlit
map_data = st_folium(mappa, width=1000, height=550)


# Controllo se l'utente ha selezionato un marker
#Per il momento mette solamente il plot del trend mensile.
if map_data and map_data.get('last_object_clicked_tooltip') != None:
    hotel_selezionato = map_data.get('last_object_clicked_tooltip')
    st.markdown(
        ''' 
        # Informazioni 
        
        '''
    )
    with st.spinner(f"Generazione del trend mensile per {hotel_selezionato}..."):
        st.write(f"**Trend Mensile per {hotel_selezionato}:**")
        plt = spark.queryManager.trend_mensile(hotel_selezionato)
        st.pyplot(plt)
    with st.spinner(f"Individuazione delle recensioni anomale per {hotel_selezionato}..."):
        extreme_reviews = spark.queryManager.anomaly_detection(hotel_selezionato)
        st.write(f"**Recensioni anomale rispetto alla media per {hotel_selezionato}:**")
        st.dataframe(extreme_reviews.toPandas())    
    with st.spinner(f"Confronto con hotel vicini a {hotel_selezionato}..."):
        hotel_lat = map_data.get("last_object_clicked").get("lat")
        hotel_lng = map_data.get("last_object_clicked").get("lng")
        nearby_hotels = spark.queryManager.get_nearby_hotels(hotel_lat,hotel_lng)
        plt = spark.queryManager.trend_mensile_compare(nearby_hotels)
        st.pyplot(plt)
        