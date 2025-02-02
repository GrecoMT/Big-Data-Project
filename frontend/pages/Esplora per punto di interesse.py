from backend import SparkBuilder
import streamlit as st
import pandas as pd

import folium
from streamlit_folium import st_folium

from pyspark.sql.functions import lit, col


st.set_page_config(page_title="Esplora per punto di interesse", layout="wide")

#Coordinate città
city_coords = {
    "Milan": [45.4642, 9.1900],
    "Vienna": [48.2082, 16.3738],
    "Barcelona": [41.3851, 2.1734],
    "London": [51.5074, -0.1278],
    "Paris": [48.8566, 2.3522],
    "Amsterdam" : [52.3709, 4.8902]
}

def getSpark(appName):
    return SparkBuilder("appName")

def hotelVicini():
    nearby_collected = nearby_hotels.collect()
    result = None
    for row in nearby_collected:
        nome = row['hotel_name']
        tmp = spark.queryManager.reputation_analysis_single(nome)
        distance = row['distance']
        tmp = tmp.withColumn("distance", lit(distance))
        if result is None:
            result = tmp
        else:
            result = result.unionByName(tmp)
    result = result.select("hotel_name", "avg_historical_score", "avg_recent_score", "distance")
    result = result.withColumn("distance", col("distance").cast("float"))
    result = result.withColumnRenamed("avg_historical_score", "Media recensioni")
    result = result.withColumnRenamed("avg_recent_score", "Media ultime recensioni")
    #st.dataframe(result.toPandas())
    return result.toPandas()


spark = getSpark("BigData_App")

if "pois" not in st.session_state:
    st.session_state.pois = pd.read_csv("Others/PointOfInterest.csv", delimiter=";")


st.title("Esplora per per punto di interesse")

st.write("Seleziona una città, dopodiché seleziona un monumento e verranno mostrati gli hotel (e le relative statistiche) vicini al punto di interesse scelto.")

city = st.selectbox(label="Seleziona una città", options=city_coords.keys(), placeholder="Città", index=None)

df = st.session_state.pois
poi_in_city = df[df['City']==city]

poi = st.selectbox(label="Seleziona un punto di interesse", options=poi_in_city, placeholder="Punto di interesse", index=None)

if city != None and poi != None:
    poi_row = df[df['Monument']==poi]
    #lat = float(poi_row["lat"])
    #lng = float(poi_row["long"])
    lat = float(poi_row["lat"].iloc[0])
    lng = float(poi_row["long"].iloc[0])

    #st.write(f"Città scelta: {city}, monumento scelto: {poi}. Latitudine poi: {lat}. Longitudine poi: {lng}")

    st.markdown("""
    ### 🏨 **Hotel vicini al punto di interesse**
    - 🟢 **Cerchio Verde**: Hotel entro **500 metri** dal punto di interesse scelto.
    - 🔵 **Cerchio Blu**: Hotel tra **500 e 1000 metri** dal punto di interesse scelto.
    - 🔴 **Cerchio Rosso**: Hotel tra **1000 e 2000 metri** dal punto di interesse scelto.
    """)

    with st.spinner("Individuo gli hotel vicini al punto di interese..."):
        nearby_hotels = spark.queryManager.get_nearby_hotels(lat, lng, 2000).select("hotel_name", "lat", "lng", "distance").distinct()
        nearby_hotels_p = nearby_hotels.toPandas()

        # Creare una mappa con Folium
        mappa = folium.Map(location=city_coords[city], zoom_start=12)

        #Aggiungere marker per ogni hotel
        for _, row in nearby_hotels_p.iterrows():
            folium.Marker(
                location=[row['lat'], row['lng']],
                tooltip=row['hotel_name'],
                icon=folium.Icon(color='blue', icon='info-sign'),
            ).add_to(mappa)

        folium.Marker(
            location=[lat, lng],
            icon = folium.Icon(color='red', tooltip=f"POI: {poi}", icon='info-sign')          
                    ).add_to(mappa)
        
        radiuses = [(500, "green"), (1000, "blue"), (2000, "red")] #raggi in metri
        for r in radiuses:
            folium.Circle(
                location=[lat, lng],
                radius=r[0],
                color=r[1],
                fill=False
            ).add_to(mappa)
        
        st_folium(mappa, width=1000, height=550)
    
    with st.spinner("Individuo i migliori hotel..."):
        # Scrivi un messaggio formattato con st.markdown
        st.markdown("""
        ### Hotel selezionati

        Stai visualizzando gli **hotel più vicini** al punto di interesse scelto, con le seguenti informazioni:

        - **Valore medio delle recensioni**: Indica la valutazione complessiva basata sulle recensioni degli utenti.
        - **Media delle ultime recensioni**: Mostra la media delle recensioni più recenti per ciascun hotel, dando un'indicazione più aggiornata sulla qualità del servizio.
        - **Distanza dal punto di interesse**: Indica la distanza in metri tra ogni hotel e il punto di interesse selezionato.

        Queste informazioni ti aiuteranno a fare una scelta più informata sugli hotel da esplorare!
        """)
        st.dataframe(hotelVicini())

