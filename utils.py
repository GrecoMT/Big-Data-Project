import requests
from urllib.parse import quote
from pyspark.sql.functions import explode, col,count, desc
import plotly.express as px

import math
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd

from nltk.corpus import wordnet

def fix_suspicious_spaces(address):
    corrections = {
        "Damr mont": "Damrémont",
        "Mont martre": "Montmartre",
        "St Ger main": "St Germain",
        "P pini re":"Pépinière",
        "arr":"Arrondissement",
        "Ga t":"Gaite",
        "Bail n":"Bailen",
        "Gr nentorgasse 30 09":"30 Grünentorgasse",
        "Landstra er G rtel 5 03":"5 Landstrasser Guertel",
        "Landstra e":"Landstraße",
        "Hasenauerstra e 12 19":"12 Hasenauerstrasse",
        "D bling":"Döbling",
        "Josefst dter Stra e 22 08":"22 Josefstädter Straße",
        "Josefst dter Stra e 10 12 08":"10 Josefstädter Straße",
        "10 12 08 Josefstadt":"Josefstadt",
        "Paragonstra e 1 11": "1 Paragonstrasse",
        "Sieveringer Stra e 4 19":"4 Sieveringer Straße Lower Sievering",
        "Pau Clar s":"Pau Claris",
        "Sep lveda":"Sepulveda",
        "Savoyenstra e 2 16":"2 Savoyenstraße",
        "W hringer Stra e 33 35 09": "33-35 Währinger Straße",
        "W hringer Stra e 12 09": "12 Währinger Straße",
        "Taborstra e 8 A 02":"8 Taborstraße"
        }
    for wrong, correct in corrections.items():
        if wrong in address:
            address = address.replace(wrong, correct)
    return address

def get_coordinates_osm(address):
    """
    Ottiene latitudine e longitudine da OpenStreetMap (Nominatim) dato un indirizzo.
    """
    try:
        #address = normalize_address(address)
        address = fix_suspicious_spaces(address)
        
        url = f"https://nominatim.openstreetmap.org/search?q={quote(address)}&format=json&addressdetails=1&limit=1"
        headers = {'User-Agent': 'Geocoding Script'}
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            if data:  # Se ci sono risultati
                location = data[0]
                lat = float(location['lat'])
                lon = float(location['lon'])
                return lat, lon
    except Exception as e:
        print(f"Errore durante il recupero delle coordinate: {e}")
    return None, None

# Funzione per ottenere latitudine
def get_lat(address):
    lat, _ = get_coordinates_osm(address)
    return lat

# Funzione per ottenere longitudine
def get_lng(address):
    _, lng = get_coordinates_osm(address)
    return lng

def get_season(month):
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Autumn"

def graficoTrend(dataframe, single : bool):
    trend_pandas = dataframe.toPandas()
    # Impostare stile del grafico
    sns.set(style="whitegrid")

    # Creare il grafico per ogni hotel
    plt.figure(figsize=(12, 6))
    for hotel in trend_pandas["Hotel_Name"].unique():
        # Filtrare i dati per ogni hotel
        hotel_data = trend_pandas[trend_pandas["Hotel_Name"] == hotel]
        
        # Creare il grafico a linee per l'hotel
        sns.lineplot(data=hotel_data, x="YearMonth", y="Average_Score", marker="o", label=hotel)

    # Migliorare l'estetica del grafico
    if single:
        plt.title("Trend dello Score Medio per Mese", fontsize=16)
    else:
        plt.title("Trend dello Score Medio per Mese degli hotel vicini", fontsize=16)
    plt.xlabel("Anno/Mese", fontsize=12)
    plt.ylabel("Punteggio Medio", fontsize=12)
    plt.xticks(rotation=45)
    plt.legend(title="Hotel", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()

    # Mostrare il grafico
    #plt.show()
    #Return necessario per inserire il grafico nel frontend
    return plt 

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])  # Converti in radianti
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a)) * 1000  # Converti in metri
 
def plot_pie_chart(df, city):
    """
    Genera un pie chart per mostrare la distribuzione dei tag in una città.
    """
    # Filtra il DataFrame per la città selezionata
    city_df = df.filter(col("Hotel_Address").contains(city))

    # Esplodi i tag (ARRAY<STRING> -> righe singole)
    tags_df = city_df.select(explode(col("tags")).alias("tag"))
 
    # Conta i tag
    tag_counts = tags_df.groupBy("tag").agg(count("*").alias("count")).orderBy(desc("count"))
    
    # Converti in Pandas per Plotly
    tag_counts_pd = tag_counts.toPandas()
    
    top_10 = tag_counts_pd[:10]
    
    others_count = tag_counts_pd[10:]["count"].sum()
    
    if others_count > 0:
        #top_10 = top_10.append({"tag": "Other", "count": others_count}, ignore_index=True)
        top_10 = pd.concat([top_10, pd.DataFrame([{"tag": "Other", "count": others_count}])], ignore_index=True)

    fig = px.pie(top_10, values="count", names="tag", title=f"Distribuzione dei tag per {city}")
    return fig

def is_adjective_or_adverb(word):
    """
    Determina se una parola è un aggettivo (a) o un avverbio (r) utilizzando WordNet.
    """
    synsets = wordnet.synsets(word)
    if not synsets:
        return False
    return any(s.pos() == 'a' or s.pos()=='r' for s in synsets)