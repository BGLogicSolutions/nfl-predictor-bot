import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib

# =====================================================================
# 1. CARGAR DATOS DE AMBAS TEMPORADAS (2024 y 2025-2026)
# =====================================================================
archivo_2024 = "temporada_2024_crudo.csv"
archivo_2025_2026 = "temporada_2025_2026_crudo.csv"

if not os.path.exists(archivo_2024):
    print(f"❌ Error: No se encontró el archivo '{archivo_2024}'.")
    exit(1)

if not os.path.exists(archivo_2025_2026):
    print(f"❌ Error: No se encontró el archivo '{archivo_2025_2026}'.")
    exit(1)

print(f"📖 Cargando datos de temporada 2024 desde {archivo_2024}...")
df_2024 = pd.read_csv(archivo_2024)

print(f"📖 Cargando datos de temporada 2025-2026 desde {archivo_2025_2026}...")
df_2025_2026 = pd.read_csv(archivo_2025_2026)

# =====================================================================
# 2. PREPARAR Y COMBINAR DATOS
# =====================================================================
# Normalizar columnas de df_2024 (asegurar que tenga las columnas necesarias)
df_2024 = df_2024[df_2024['estado_partido'] == 'FT'].copy()
df_2024 = df_2024[['semana', 'equipo_local', 'equipo_visitante', 'puntos_local', 'puntos_visitante']]

# df_2025_2026 ya tiene los datos con estado 'FT' implícito, solo seleccionar columnas necesarias
df_2025_2026 = df_2025_2026[['semana', 'equipo_local', 'equipo_visitante', 'puntos_local', 'puntos_visitante']]

# Combinar ambos DataFrames
df = pd.concat([df_2024, df_2025_2026], ignore_index=True)
df = df.sort_values(by='semana').reset_index(drop=True)

print(f"✅ Datos combinados exitosamente")
print(f"🏈 Total de partidos procesados: {len(df)}")
print(f"   - Temporada 2024: {len(df_2024)} partidos")
print(f"   - Temporada 2025-2026: {len(df_2025_2026)} partidos")

# =====================================================================
# 3. INGENIERÍA DE CARACTERÍSTICAS (PROMEDIOS MÓVILES ACUMULADOS)
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

print("\n🛠️ Calculando promedios móviles acumulados semana a semana...")
print("   (Utilizando datos de TODAS las temporadas en el historial)")

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
# 4. DIVISIÓN INTELIGENTE (EVITANDO DATA LEAKAGE)
# =====================================================================
# Estrategia de entrenamiento mejorada:
# - Entrenamos con el 80% de los datos históricos (combinados de ambas temporadas)
# - Evaluamos con el 20% restante (datos más recientes)

total_partidos = len(df)
punto_corte = int(0.8 * total_partidos)

df_train = df.iloc[:punto_corte]
df_test = df.iloc[punto_corte:]

caracteristicas = ['prom_puntos_local_anotados', 'prom_puntos_local_recibidos', 
                   'prom_puntos_vis_anotados', 'prom_puntos_vis_recibidos']

X_train = df_train[caracteristicas]
y_train = df_train['gana_local']

X_test = df_test[caracteristicas]
y_test = df_test['gana_local']

print(f"\n📊 División de datos para entrenamiento:")
print(f"   - Partidos para entrenamiento: {len(X_train)} ({len(X_train)/total_partidos*100:.1f}%)")
print(f"   - Partidos para evaluación: {len(X_test)} ({len(X_test)/total_partidos*100:.1f}%)")

# =====================================================================
# 5. ESCALADO DE VARIABLES Y ENTRENAMIENTO
# =====================================================================
escalador = StandardScaler()
X_train_escalado = escalador.fit_transform(X_train)
X_test_escalado = escalador.transform(X_test)

modelo = LogisticRegression(max_iter=1000, random_state=42)
modelo.fit(X_train_escalado, y_train)
print("\n🎯 ¡Modelo de Regresión Logística entrenado exitosamente!")
print(f"   Datos de entrenamiento: 2024 + 2025-2026 (combinados)")

# =====================================================================
# 6. EVALUACIÓN DEL RENDIMIENTO DEL MODELO
# =====================================================================
predicciones = modelo.predict(X_test_escalado)
precision = accuracy_score(y_test, predicciones)

print(f"\n📈 ==================================================")
print(f"   RENDIMIENTO DEL MODELO (DATOS COMBINADOS)")
print(f"   Precisión Global (Accuracy): {precision:.2%}")
print(f"==================================================\n")
print("Reporte de Clasificación detallado:")
print(classification_report(y_test, predicciones, target_names=['Gana Visitante', 'Gana Local']))

# =====================================================================
# 7. GUARDAR MODELO Y ESCALADOR
# =====================================================================
joblib.dump(modelo, 'modelo_predicciones.pkl')
joblib.dump(escalador, 'escalador_predicciones.pkl')
print("\n💾 Modelo y escalador guardados exitosamente")
print("   - modelo_predicciones.pkl")
print("   - escalador_predicciones.pkl")

# =====================================================================
# 8. RESUMEN Y PRÓXIMOS PASOS
# =====================================================================
print(f"\n✅ RESUMEN DEL REENTRENAMIENTO:")
print(f"   • Datos utilizados: {len(df)} partidos totales")
print(f"   • Fuentes: Temporada 2024 + Temporada 2025-2026")
print(f"   • Historial de equipos: ACTUALIZADO con datos recientes")
print(f"   • Modelo guardado: LISTO para realizar predicciones")
print(f"\n🚀 El modelo está optimizado y listo para usar!")
