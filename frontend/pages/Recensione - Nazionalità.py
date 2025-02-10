import streamlit as st
import pandas as pd
import pydeck as pdk
from pyspark.sql.functions import col
from geopy.geocoders import Nominatim
import time
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable

st.set_page_config(page_title="🌍 Analisi Recensioni per Nazionalità", layout="wide")

# Cache per Spark e dati
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

@st.cache_data
def nationality_review_analysis_cached(min_reviews):
    return spark.queryManager.nationality_review_analysis(min_reviews=min_reviews).toPandas()

@st.cache_data
def get_reviews_for_nationality(nationality):
    df = spark.queryManager.get_reviews_for_nationality(nationality)
    return df.toPandas()

@st.cache_data
def get_coordinates(nationalities, attempt=1, max_attempts=5):
    geolocator = Nominatim(user_agent="geoapi", timeout=10) 
    coordinates = {}
    for nationality in nationalities:
        try:
            location = geolocator.geocode(nationality)
            if location:
                coordinates[nationality] = [location.latitude, location.longitude]
            else:
                coordinates[nationality] = None
        except (GeocoderTimedOut, GeocoderUnavailable):
            if attempt < max_attempts:
                time.sleep(2)  
                return get_coordinates(nationalities, attempt + 1, max_attempts)
            else:
                coordinates[nationality] = None
                st.warning(f"Impossibile ottenere coordinate per {nationality}.")
        time.sleep(1)  # Ritardo di 1 secondo tra le richieste
    return coordinates

# Parametri per la mappa
st.title("🌍 Analisi Recensioni per Nazionalità")
nationality_reviews = nationality_review_analysis_cached(1) #serve per la mappa

# Ottieni le coordinate per le nazionalità
unique_nationalities = nationality_reviews["Reviewer_Nationality"].unique()
if "coordinates" not in st.session_state.to_dict():
        st.session_state.coordinates = get_coordinates(unique_nationalities)

# Prepara il DataFrame per la mappa
map_data = nationality_reviews.copy()
map_data["coordinates"] = map_data["Reviewer_Nationality"].map(st.session_state.coordinates)
map_data = map_data.dropna(subset=["coordinates"])
map_data["lat"] = map_data["coordinates"].apply(lambda x: x[0])
map_data["lon"] = map_data["coordinates"].apply(lambda x: x[1])

# Mappa con i puntini
st.subheader("Mappa delle nazionalità dei recensori")
map_layer = pdk.Layer(
    "ScatterplotLayer",
    data=map_data,
    get_position=["lon", "lat"],
    get_radius=50000,
    get_fill_color=[0, 128, 255, 200],  # Blu trasparente (mi fa un po sboccare)
    pickable=False,
)

r = pdk.Deck(
    layers=[map_layer],
    initial_view_state=pdk.ViewState(latitude=0, longitude=0, zoom=1),
    tooltip={"text": "Nazionalità: {Reviewer_Nationality}\nPunteggio medio: {avg_score}"}
)

st.pydeck_chart(r)
st.write("---")

min_reviews = st.selectbox("Seleziona il numero minimo di recensioni per includere una nazionalità", [1, 100, 1000])

st.markdown("### Dati delle recensioni per nazionalità")

# Selettore per la nazionalità
selected_nationality = st.selectbox(
    "Seleziona una nazionalità:",
    options=nationality_review_analysis_cached(min_reviews),
    index=0
)

# Analisi basata sulla selezione
if selected_nationality:
    st.markdown(f"### Analisi per {selected_nationality}")

    # Carica i dati delle recensioni per la nazionalità selezionata
    with st.spinner("Caricamento recensioni..."):
        reviews = get_reviews_for_nationality(selected_nationality)

    if not reviews.empty:
        st.write("📊 Dati delle recensioni:")
        st.dataframe(reviews)

        # Grafici delle recensioni
        st.write("📈 Conteggio recensioni non sufficienti e sufficienti:")
        st.bar_chart(reviews["Reviewer_Score"].apply(lambda x: "Negative (<6)" if x < 6 else "Positive (≥6)").value_counts())
    else:
        st.warning("Nessuna recensione trovata per questa nazionalità.")
else:
    st.info("Seleziona una nazionalità per visualizzare i dettagli.")
