import os
import requests
import pandas as pd

# =====================================================================
# 1. AUTENTICACIÓN CON LA API
# =====================================================================
API_KEY = os.environ.get("API_SPORTS_KEY")

if not API_KEY:
    print("❌ Error: No se encontró la variable de entorno 'API_SPORTS_KEY'.")
    print("Asegúrate de haber guardado tu API key en Settings > Secrets > Actions.")
    exit(1)

# Endpoint oficial de API-Sports para Fútbol Americano (NFL)
URL = "https://v1.american-football.api-sports.io/games"
HEADERS = {
    "x-apisports-key": API_KEY
}
PARAMS = {
    "league": "1",    # ID 1 corresponde a la NFL
    "season": "2024"   # Temporada completa disponible en el plan gratuito
}

# =====================================================================
# 2. PETICIÓN Y DESCARGA
# =====================================================================
print("🚀 Conectando con API-Sports para descargar la temporada NFL 2024...")
try:
    response = requests.get(URL, headers=HEADERS, params=PARAMS)
    response.raise_for_status()  
    data = response.json()
except Exception as e:
    print(f"❌ Error crítico en la conexión: {e}")
    exit(1)

# Validamos que la API no haya devuelto un mensaje de error interno
if "errors" in data and data["errors"]:
    print(f"❌ La API devolvió un error: {data['errors']}")
    exit(1)

partidos_raw = data.get("response", [])

if not partidos_raw:
    print("⚠️ No se encontraron partidos para los parámetros establecidos.")
    exit(0)

print(f"📦 Datos descargados con éxito. Se encontraron {len(partidos_raw)} registros.")

# =====================================================================
# 3. LIMPIEZA Y ESTRUCTURACIÓN DE DATOS (DATA WRANGLING)
# =====================================================================
registros_limpios = []

for partido in partidos_raw:
    game_info = partido.get("game", {})
    teams_info = partido.get("teams", {})
    scores_info = partido.get("scores", {})
    
    # Extraer el número de la semana eliminando el texto (ej: "Regular Season - Week 5" -> 5)
    semana_texto = game_info.get("week", "Week 1")
    try:
        semana = int(''.join(filter(str.isdigit, semana_texto)))
    except ValueError:
        semana = 1

    registros_limpios.append({
        "id_partido": game_info.get("id"),
        "fecha": game_info.get("date"),
        "semana": semana,
        "estadio": game_info.get("venue", {}).get("name"),
        "estado_partido": game_info.get("status", {}).get("short"), # "FT" significa terminado
        "equipo_local": teams_info.get("home", {}).get("name"),
        "equipo_visitante": teams_info.get("away", {}).get("name"),
        "puntos_local": scores_info.get("home", {}).get("total"),
        "puntos_visitante": scores_info.get("away", {}).get("total")
    })

# Convertimos la lista de diccionarios en un DataFrame de Pandas
df_temporada = pd.DataFrame(registros_limpios)

# Ordenamos cronológicamente por número de semana
df_temporada = df_temporada.sort_values(by="semana").reset_index(drop=True)

# =====================================================================
# 4. ALMACENAMIENTO LOCAL
# =====================================================================
archivo_salida = "temporada_2024_crudo.csv"
df_temporada.to_csv(archivo_salida, index=False)

print(f"\n💾 ¡Proceso finalizado! Archivo '{archivo_salida}' guardado correctamente.")
print(f"📊 Resumen de las primeras filas descargadas:\n")
print(df_temporada[['semana', 'equipo_local', 'puntos_local', 'equipo_visitante', 'puntos_visitante']].head())
