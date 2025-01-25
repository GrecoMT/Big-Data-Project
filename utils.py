import requests
from urllib.parse import quote

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