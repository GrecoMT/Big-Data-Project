from backend import SparkBuilder

import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="Hotel Map", layout="wide")

@st.cache_resource
def getSpark(appName):
    return SparkBuilder("appName", "/Users/matteog/Documents/Università/Laurea Magistrale/Big Data/Progetto/Dataset/Hotel_Reviews.csv")

spark = getSpark("BigData_App")

dataframe = spark.df_finale

df_temp = dataframe.select("hotel_name", "hotel_address", "lat", "lng").distinct().cache()

df = df_temp.toPandas()

# Configurare la pagina di Streamlit

st.title("Mappa degli Hotel")

# Creare una mappa con Folium
mappa = folium.Map(location=[df['lat'].mean(), df['lng'].mean()], zoom_start=13)

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
    st.write(f"**Trend Mensile per {hotel_selezionato}:**")
    plt = spark.queryManager.trend_mensile(hotel_selezionato)
    st.pyplot(plt)
