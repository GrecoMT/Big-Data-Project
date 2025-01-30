from backend import SparkBuilder
import streamlit as st
import utils

st.set_page_config(page_title="Hotel Map", layout="wide")

@st.cache_resource
def getSpark(appName):
    return SparkBuilder("appName")

spark = getSpark("BigData_App")

tags = utils.get_most_used_tags(spark.df_finale)

st.title("Esplora per tag")
#st.markdown("# Seleziona una città:")
st.sidebar.title("🔍 Navigazione")
st.sidebar.markdown("### Sezioni disponibili:")

st.sidebar.markdown("- 🏠 **Home**")
st.sidebar.markdown("- 📍 **Mappa Hotel**")
st.sidebar.markdown("- 📊 **Trend & Analisi**")
st.sidebar.markdown("- 🔍 **Anomaly Detection**")
st.sidebar.markdown("- 📝 **Word Cloud**")

#Coordinate città
city_coords = {
    "Milano": [45.4642, 9.1900],
    "Vienna": [48.2082, 16.3738],
    "Barcellona": [41.3851, 2.1734],
    "Londra": [51.5074, -0.1278],
    "Parigi": [48.8566, 2.3522]
}

option = st.selectbox(label = "Seleziona un tag", options=tags, index = None, placeholder="Tag")

col1, col2, col3, col4, col5 = st.columns(5)

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
    
city = None  
    
if button1:
    city = "Milan"
elif button2:
    city = "Vienna"
elif button3:
    city = "Barcelona"
elif button4:
    city = "London"
elif button5:
    city = "Paris"


if city != None:
    st.title(f"Migliori hotel con tag {option} in {city}")
    print(f"TAG {option}, CITTA {city}")
    result = spark.queryManager.get_hotels_by_tag(city=city, tag=option)
    if result is not None:
        st.table(result.toPandas())