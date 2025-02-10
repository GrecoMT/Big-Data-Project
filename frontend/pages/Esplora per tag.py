from backend import SparkBuilder
import streamlit as st
import utils

from pyspark.sql.functions import asc

st.set_page_config(page_title="Esplora per tag", layout="wide")

@st.cache_resource
def getSpark(appName):
    return SparkBuilder(appName)

spark = getSpark("BigData_App")

tags = spark.queryManager.get_most_used_tags()

st.title("Esplora per tag")

st.sidebar.title("🔍 Navigazione")
st.sidebar.markdown("### Sezioni disponibili:")

st.sidebar.markdown("- 🏠 **Home**")
st.sidebar.markdown("- 📊 **Analisi delle parole**")
st.sidebar.markdown("- 🗺️ **Esplora con mappa**")
st.sidebar.markdown("- 📍 **Esplora per punto di interesse**")
st.sidebar.markdown("- #️⃣ **Esplora per tag**")
st.sidebar.markdown("- 🇮🇹 **Recensione-Nazionalità**")
st.sidebar.markdown("- 🏖️ **Sentiment Stagionale**")

#Coordinate città
city_coords = {
    "Milano": [45.4642, 9.1900],
    "Vienna": [48.2082, 16.3738],
    "Barcellona": [41.3851, 2.1734],
    "Londra": [51.5074, -0.1278],
    "Parigi": [48.8566, 2.3522],
    "Amsterdam" : [52.3709, 4.8902]
}

@st.cache_data
def tag_influence_asc():
    df = spark.queryManager.tag_influence_analysis().orderBy(asc("avg_score")).limit(10)
    return df.toPandas()

@st.cache_data
def tag_influence_desc():
    df = spark.queryManager.tag_influence_analysis().limit(10)
    return df.toPandas()

st.subheader("Influenza tag sullo scoring")

col1, col2 = st.columns(2)
with col1:
    st.write("Tag con score migliore:")
    st.table(tag_influence_desc())

with col2:
    st.write("Tag con score peggiore:")
    st.table(tag_influence_asc())

st.markdown("## Seleziona una città:")

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


if st.session_state.selected_button: 
    pie_plot = utils.plot_pie_chart(spark.df_finale,st.session_state.selected_button)
    st.plotly_chart(pie_plot)
    
    if "selected_option" not in st.session_state:
        st.session_state.selected_option = None
    option = st.selectbox(label = "Seleziona un tag", options=tags, index=None, placeholder="Tag")
    if option != None:
        st.session_state.selected_option = option
        with st.spinner(f"Individuazione dei migliori hotel in {st.session_state.selected_button} con tag {option}..."):
            st.markdown(
                "<h4>Hotel ordinati secondo lo score medio</h4>"
                "<p>Tabella contenente gli hotel della città selezionata "
                "associati al tag selezionato, ordinati in modo decrescente.</p>",
                unsafe_allow_html=True
            )
            result = spark.queryManager.get_hotels_by_tag(city=st.session_state.selected_button, tag=option)
            if result is not None:
                st.table(result.toPandas())
        
        