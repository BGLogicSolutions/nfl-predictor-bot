import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# =====================================================================
# 1. CARGAR DATOS DE LA TEMPORADA 2024
# =====================================================================
archivo_entrada = "temporada_2024_crudo.csv"

if not os.path.exists(archivo_entrada):
    print(f"❌ Error: No se encontró el archivo '{archivo_entrada}'.")
    print("Asegúrate de haber ejecutado primero el paso anterior de descarga.")
    exit(1)

print(f"📖 Cargando datos históricos desde {archivo_entrada}...")
df = pd.read_csv(archivo_entrada)

# Filtramos solo los partidos terminados ("FT") y los ordenamos por semana
df = df[df['estado_partido'] == 'FT'].sort_values(by='semana').reset_index(drop=True)
print(f"🏈 Procesando {len(df)} partidos finalizados de la temporada regular.")

# =====================================================================
# 2. INGENIERÍA DE CARACTERÍSTICAS (PROMEDIOS MÓVILES ACUMULADOS)
# =====================================================================
# Creamos diccionarios para registrar el desempeño de cada equipo paso a paso
equipos = pd.concat([df['equipo_local'], df['equipo_visitante']]).unique()
historial_anotados = {equipo: [] for equipo in equipos}
historial_recibidos = {equipo: [] for equipo in equipos}

# Inicializamos las características con 20 puntos (promedio estándar de la NFL)
df['prom_puntos_local_anotados'] = 20.0
df['prom_puntos_local_recibidos'] = 20.0
df['prom_puntos_vis_anotados'] = 20.0
df['prom_puntos_vis_recibidos'] = 20.0

print("🛠️ Calculando promedios móviles acumulados semana a semana...")
for idx, row in df.iterrows():
    loc = row['equipo_local']
    vis = row['equipo_visitante']
    
    # Asignamos el promedio que tenía el equipo ANTES de jugar este partido
    if historial_anotados[loc]:
        df.at[idx, 'prom_puntos_local_anotados'] = np.mean(historial_anotados[loc])
        df.at[idx, 'prom_puntos_local_recibidos'] = np.mean(historial_recibidos[loc])
    if historial_anotados[vis]:
        df.at[idx, 'prom_puntos_vis_anotados'] = np.mean(historial_anotados[vis])
        df.at[idx, 'prom_puntos_vis_recibidos'] = np.mean(historial_recibidos[vis])
        
    # Guardamos el resultado real de este partido para usarlo en las semanas siguientes
    historial_anotados[loc].append(row['puntos_local'])
    historial_recibidos[loc].append(row['puntos_visitante'])
    
    historial_anotados[vis].append(row['puntos_visitante'])
    historial_recibidos[vis].append(row['puntos_local'])

# Definimos la variable objetivo (Target): 1 = Ganó Local, 0 = Ganó Visitante
df['gana_local'] = (df['puntos_local'] > df['puntos_visitante']).astype(int)

# =====================================================================
# 3. DIVISIÓN CRONOLÓGICA (EVITANDO DATA LEAKAGE)
# =====================================================================
# Entrenamos con el grueso de la temporada (Semanas 1 a 14)
# Evaluamos la precisión con el cierre de la temporada (Semanas 15 a 18)
df_train = df[df['semana'] <= 14]
df_test = df[df['semana'] > 14]

caracteristicas = ['prom_puntos_local_anotados', 'prom_puntos_local_recibidos', 
                   'prom_puntos_vis_anotados', 'prom_puntos_vis_recibidos']

X_train = df_train[caracteristicas]
y_train = df_train['gana_local']

X_test = df_test[caracteristicas]
y_test = df_test['gana_local']

print(f"📊 Partidos para entrenamiento (Semanas 1-14): {len(X_train)}")
print(f"📊 Partidos para evaluación (Semanas 15-18): {len(X_test)}")

# =====================================================================
# 4. ESCALADO DE VARIABLES Y ENTRENAMIENTO
# =====================================================================
escalador = StandardScaler()
X_train_escalado = escalador.fit_transform(X_train)
X_test_escalado = escalador.transform(X_test)

modelo = LogisticRegression()
modelo.fit(X_train_escalado, y_train)
print("🎯 ¡Modelo de Regresión Logística entrenado con éxito!")

# =====================================================================
# 5. EVALUACIÓN DEL RENDIMIENTO DEL MODELO
# =====================================================================
predicciones = modelo.predict(X_test_escalado)
precision = accuracy_score(y_test, predicciones)

print(f"\n📈 ==================================================")
print(f"   RENDIMIENTO DEL MODELO EN SEMANAS RESTRINGIDAS (15-18)")
print(f"   Precisión Global (Accuracy): {precision:.2%}")
print(f"==================================================\n")
print("Reporte de Clasificación detallado:")
print(classification_report(y_test, predicciones, target_names=['Gana Visitante', 'Gana Local']))
