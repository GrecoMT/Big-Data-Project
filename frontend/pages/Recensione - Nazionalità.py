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
st.sidebar.markdown("- 🗺️ **Esplora con mappa**")
st.sidebar.markdown("- 📍**Esplora per punto di interesse**")
st.sidebar.markdown("- #️⃣ **Esplora per tag**")
st.sidebar.markdown("- 🇮🇹 **Recensione-Nazionalità**")
st.sidebar.markdown("- 🏖️ **Sentiment Stagionale**")

@st.cache_data
def nationality_review_analysis_cached(n, min_reviews):
    return spark.queryManager.nationality_review_analysis(n=n, min_reviews=min_reviews).toPandas()

def get_reviews_for_nationality(nationality):
    # Filtra i dati per la nazionalità selezionata
    if nationality:
        return spark.df_finale.filter(col("Reviewer_Nationality") == nationality).select("Hotel_Name", "Review_Date", "Reviewer_Nationality", "Reviewer_Score","Negative_Review", "Review_Total_Negative_Word_Counts", "Positive_Review", "Review_Total_Positive_Word_Counts",  "Tags", ).toPandas()
    return pd.DataFrame()  # Restituisce un DataFrame vuoto se non è selezionata una nazionalità

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
                return get_coordinates(nationalities, attempt + 1, max_attempts) #RICORSIONE SI SI SI SI SI SI SI SI SI SI SI SI SI SI SI SI SI SI SI SI SI
            else:
                coordinates[nationality] = None
                st.warning(f"Impossibile ottenere coordinate per {nationality}.")
        time.sleep(1)  # Ritardo di 1 secondo tra le richieste
    return coordinates

# Parametri per la mappa
st.title("🌍 Analisi Recensioni per Nazionalità")
n = 100
min_reviews = st.slider("Seleziona il numero minimo di recensioni per includere una nazionalità", 1, 50, 100)

st.markdown("### Dati delle recensioni per nazionalità")
nationality_reviews = nationality_review_analysis_cached(n, min_reviews)

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
st.subheader("🌍 Mappa delle Nazionalità")
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

# Selettore per la nazionalità
selected_nationality = st.selectbox(
    "Seleziona una nazionalità:",
    options=map_data["Reviewer_Nationality"].unique(),
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
