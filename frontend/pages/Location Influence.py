import streamlit as st
from spark_builder import SparkBuilder
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import folium
from query_manager import QueryManager
from streamlit_folium import st_folium

@st.cache_resource
def getSpark(appName):
    return SparkBuilder("appName", "/Users/matteog/Documents/Università/Laurea Magistrale/Big Data/Progetto/Dataset/Hotel_Reviews.csv")

spark = getSpark("BigData_App")
#st.set_page_config(page_title="QUERY CLUSTER", page_icon="📈")

st.subheader("Geographic Locations")
st.markdown("La mappa mostra i cluster geografici degli hotel.")
query = QueryManager(spark)
map_ = query.location_influence()
st_folium(map_, width=1000)