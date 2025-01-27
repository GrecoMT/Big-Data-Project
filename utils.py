import requests
from urllib.parse import quote
import math
import numpy as np

#import per prova grafico trend.
import matplotlib.pyplot as plt
import seaborn as sns

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


#----CLUSTERING UTILS FUNC----------
def calculate_cluster_center(df_cluster):
    center_lat = np.mean(df_cluster["lat"])
    center_lng = np.mean(df_cluster["lng"])
    return center_lat, center_lng

def calculate_lat_lng_radius(center_lat, radius_km):
    delta_lat = radius_km / 111  # Delta latitudine
    delta_lng = radius_km / (111 * math.cos(math.radians(center_lat)))  # Delta longitudine
    return delta_lat, delta_lng

def calculate_cluster_radius(df_cluster, center_lat, center_lng):
    distances = []
    for _, row in df_cluster.iterrows():
        # Calcola la distanza Haversine in km
        lat_diff = np.radians(row["lat"] - center_lat)
        lng_diff = np.radians(row["lng"] - center_lng)
        a = np.sin(lat_diff / 2) ** 2 + np.cos(np.radians(center_lat)) * np.cos(np.radians(row["lat"])) * np.sin(lng_diff / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        distances.append(6371 * c)  # Raggio della Terra in km
    return max(distances)

#----------------------------------------------------------------

def get_season(month):
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Autumn"

def graficoTrend(dataframe):
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
    plt.title("Trend dello Score Medio per Mese (per Hotel)", fontsize=16)
    plt.xlabel("Mese", fontsize=12)
    plt.ylabel("Punteggio Medio", fontsize=12)
    plt.xticks(rotation=45)
    plt.legend(title="Hotel", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()

    # Mostrare il grafico
    plt.show()