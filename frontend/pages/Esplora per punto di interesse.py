from backend import SparkBuilder
import streamlit as st
import pandas as pd

import folium
from streamlit_folium import st_folium

from folium.plugins import TagFilterButton


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

@st.cache_data
def hotelVicini_pandas(dataframe):
    result_list = []  # Lista per accumulare i risultati

    for _, row in dataframe.iterrows():
        nome = row['hotel_name']
        distance = float(row['distance'])  # Convertiamo subito la distanza in float

        # Simuliamo la funzione Spark queryManager.reputation_analysis_single(nome)
        tmp = spark.queryManager.reputation_analysis_single(nome).toPandas()  # Deve restituire un DataFrame Pandas

        # Se il DataFrame è vuoto, passiamo al prossimo hotel
        if tmp.empty:
            continue  

        # Aggiungiamo la colonna "distance"
        tmp = tmp.copy()  # Evita SettingWithCopyWarning
        tmp["distance"] = distance

        # Accumula il risultato
        result_list.append(tmp)

    # Unisce tutti i DataFrame in uno solo
    if result_list:
        result = pd.concat(result_list, ignore_index=True)
    else:
        result = pd.DataFrame(columns=["Hotel_Name", "avg_historical_score", "avg_recent_score", "distance"])

    # Seleziona e rinomina le colonne finali
    result = result[["Hotel_Name", "avg_historical_score", "avg_recent_score", "distance"]]
    result = result.rename(columns={
        "avg_historical_score": "Media recensioni",
        "avg_recent_score": "Media ultime recensioni"
    })

    return result  # Restituisce un DataFrame Pandas

spark = getSpark("BigData_App")

st.sidebar.title("🔍 Navigazione")
st.sidebar.markdown("### Sezioni disponibili:")

st.sidebar.markdown("- 🏠 **Home**")
st.sidebar.markdown("- 🗺️ **Esplora con mappa**")
st.sidebar.markdown("- 📍**Esplora per punto di interesse**")
st.sidebar.markdown("- #️⃣ **Esplora per tag**")
st.sidebar.markdown("- 🇮🇹 **Recensione-Nazionalità**")
st.sidebar.markdown("- 🏖️ **Sentiment Stagionale**")

@st.cache_data
def nearby_hotels(lat,lng):
    nearby_hotels = spark.queryManager.get_nearby_hotels(lat, lng, 2000).select("hotel_name", "lat", "lng", "distance").distinct()
    return nearby_hotels.toPandas()

if "pois" not in st.session_state:
    st.session_state.pois = pd.read_csv("Others/PointOfInterest.csv", delimiter=";")

def categoria_distanza(distanza):
    if float(distanza) <= 500:
        return "Entro 500 metri"
    elif 500 < float(distanza) <= 1000:
        return "Tra 500 e 1000 metri"
    elif 1000 < float(distanza) <= 2000:
        return "Tra 1000 e 2000 metri"


@st.cache_data
def createMap(nearby_hotels_p, poi):

    dfp = st.session_state.pois
    result = dfp.loc[dfp["Monument"] == poi, ["lat", "long"]].values

    # Converte in array (lista Python)
    coordinate = result[0].tolist() if len(result) > 0 else None

    # Creare una mappa con Folium
    mappa = folium.Map(location=coordinate, zoom_start=13.5)

    #Aggiungere marker per ogni hotel
    for _, row in nearby_hotels_p.iterrows():
            category = categoria_distanza(row['distance'])
            folium.Marker(
                tags = [category],
                location=[row['lat'], row['lng']],
                tooltip=row['hotel_name'],
                icon=folium.Icon(color='darkblue', icon='fa-solid fa-hotel', prefix='fa'),
            ).add_to(mappa)

    folium.Marker(
            location=[lat, lng],
            tooltip=poi.upper(),
            icon = folium.Icon(color='darkred',  icon='fa-solid fa-monument', prefix='fa')          
                    ).add_to(mappa)
    
    TagFilterButton(["Entro 500 metri", "Tra 500 e 1000 metri", "Tra 1000 e 2000 metri"]).add_to(mappa)

        
    radiuses = [(500, "green"), (1000, "blue"), (2000, "red")] #raggi in metri
    for r in radiuses:
        folium.Circle(
            location=[lat, lng],
            radius=r[0],
            color=r[1],
            fill=False
        ).add_to(mappa)
    return mappa

st.title("Esplora per per punto di interesse")

st.write("Seleziona una città, dopodiché seleziona un monumento e verranno mostrati gli hotel (e le relative statistiche) vicini al punto di interesse scelto.")

city = st.selectbox(label="Seleziona una città", options=city_coords.keys(), placeholder="Città", index=None)

df = st.session_state.pois
poi_in_city = df[df['City']==city]

poi = st.selectbox(label="Seleziona un punto di interesse", options=poi_in_city, placeholder="Punto di interesse", index=None)

if city != None and poi != None:
    poi_row = df[df['Monument']==poi]
    lat = float(poi_row["lat"].iloc[0])
    lng = float(poi_row["long"].iloc[0])

    nearby_hotels = nearby_hotels(lat, lng)

    st.markdown("""
    ### 🏨 **Hotel vicini al punto di interesse**
    - 🟢 **Cerchio Verde**: Hotel entro **500 metri** dal punto di interesse scelto.
    - 🔵 **Cerchio Blu**: Hotel tra **500 e 1000 metri** dal punto di interesse scelto.
    - 🔴 **Cerchio Rosso**: Hotel tra **1000 e 2000 metri** dal punto di interesse scelto.
                
    🗺️ **Filtra gli hotel per distanza direttamente sulla mappa**:
    """)

    with st.spinner("Individuo gli hotel vicini al punto di interese..."):
        mappa = createMap(nearby_hotels, poi)
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
        st.dataframe(hotelVicini_pandas(nearby_hotels))

