import streamlit as st
from backend import SparkBuilder
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from streamlit_folium import st_folium

st.set_page_config(page_title="QUERY CLUSTER", page_icon="📈")

@st.cache_resource
def getSpark(appName):
    return SparkBuilder("appName", "/Users/matteog/Documents/Università/Laurea Magistrale/Big Data/Progetto/Dataset/Hotel_Reviews.csv")

spark = getSpark("BigData_App")

st.subheader("Geographic Locations")
st.markdown("La mappa mostra i cluster geografici degli hotel.")
query = spark.queryManager
map_ = query.location_influence()
st_folium(map_, width=1000)