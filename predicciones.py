import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import joblib

# =====================================================================
# 1. CARGAR DATOS DE AMBAS TEMPORADAS (2024 y 2025-2026)
# =====================================================================
archivo_2024 = "temporada_2024_crudo.csv"
archivo_2025_2026 = "temporada_2025_2026_crudo.csv"

if not os.path.exists(archivo_2024) or not os.path.exists(archivo_2025_2026):
    print(f"❌ Error: Falta alguno de los archivos de datos.")
    exit(1)

print(f"📖 Cargando datos de temporada 2024...")
df_2024 = pd.read_csv(archivo_2024)

print(f"📖 Cargando datos de temporada 2025-2026...")
df_2025_2026 = pd.read_csv(archivo_2025_2026)

# =====================================================================
# 2. PREPARAR Y COMBINAR DATOS
# =====================================================================
df_2024 = df_2024[df_2024['estado_partido'] == 'FT'].copy()
df_2024 = df_2024[['semana', 'equipo_local', 'equipo_visitante', 'puntos_local', 'puntos_visitante']]

df_2025_2026 = df_2025_2026[['semana', 'equipo_local', 'equipo_visitante', 'puntos_local', 'puntos_visitante']]

df = pd.concat([df_2024, df_2025_2026], ignore_index=True)
df = df.sort_values(by='semana').reset_index(drop=True)

print(f"✅ Datos combinados exitosamente")
print(f"🏈 Total de partidos procesados: {len(df)}")

# =====================================================================
# 3. INGENIERÍA AVANZADA DE CARACTERÍSTICAS
# =====================================================================
print("\n🛠️ Creando características avanzadas...")

equipos = pd.concat([df['equipo_local'], df['equipo_visitante']]).unique()
historial_anotados = {equipo: [] for equipo in equipos}
historial_recibidos = {equipo: [] for equipo in equipos}

# Características básicas
df['prom_puntos_local_anotados'] = 20.0
df['prom_puntos_local_recibidos'] = 20.0
df['prom_puntos_vis_anotados'] = 20.0
df['prom_puntos_vis_recibidos'] = 20.0
df['diferencia_puntos_local'] = 0.0
df['diferencia_puntos_vis'] = 0.0
df['varianza_anotados_local'] = 0.0
df['varianza_recibidos_local'] = 0.0
df['varianza_anotados_vis'] = 0.0
df['varianza_recibidos_vis'] = 0.0
df['games_played_local'] = 0
df['games_played_vis'] = 0

print("   Calculando promedios móviles, varianzas y tendencias...")

for idx, row in df.iterrows():
    loc = row['equipo_local']
    vis = row['equipo_visitante']
    
    # PROMEDIO MÓVIL
    if historial_anotados[loc]:
        df.at[idx, 'prom_puntos_local_anotados'] = np.mean(historial_anotados[loc])
        df.at[idx, 'prom_puntos_local_recibidos'] = np.mean(historial_recibidos[loc])
        # VARIANZA (indica consistencia)
        df.at[idx, 'varianza_anotados_local'] = np.var(historial_anotados[loc])
        df.at[idx, 'varianza_recibidos_local'] = np.var(historial_recibidos[loc])
        df.at[idx, 'games_played_local'] = len(historial_anotados[loc])
        
    if historial_anotados[vis]:
        df.at[idx, 'prom_puntos_vis_anotados'] = np.mean(historial_anotados[vis])
        df.at[idx, 'prom_puntos_vis_recibidos'] = np.mean(historial_recibidos[vis])
        df.at[idx, 'varianza_anotados_vis'] = np.var(historial_anotados[vis])
        df.at[idx, 'varianza_recibidos_vis'] = np.var(historial_recibidos[vis])
        df.at[idx, 'games_played_vis'] = len(historial_anotados[vis])
    
    # DIFERENCIA (métrica clave)
    df.at[idx, 'diferencia_puntos_local'] = df.at[idx, 'prom_puntos_local_anotados'] - df.at[idx, 'prom_puntos_local_recibidos']
    df.at[idx, 'diferencia_puntos_vis'] = df.at[idx, 'prom_puntos_vis_anotados'] - df.at[idx, 'prom_puntos_vis_recibidos']
    
    # Guardar resultados
    historial_anotados[loc].append(row['puntos_local'])
    historial_recibidos[loc].append(row['puntos_visitante'])
    
    historial_anotados[vis].append(row['puntos_visitante'])
    historial_recibidos[vis].append(row['puntos_local'])

# CARACTERÍSTICAS DERIVADAS (interacciones y ratios)
df['ratio_ofensivo'] = df['prom_puntos_local_anotados'] / (df['prom_puntos_vis_recibidos'] + 1)
df['ratio_defensivo'] = df['prom_puntos_local_recibidos'] / (df['prom_puntos_vis_anotados'] + 1)
df['ventaja_ofensiva'] = df['prom_puntos_local_anotados'] - df['prom_puntos_vis_recibidos']
df['ventaja_defensiva'] = df['prom_puntos_vis_anotados'] - df['prom_puntos_local_recibidos']
df['experiencia_relativa'] = df['games_played_local'] - df['games_played_vis']

# Variable objetivo
df['gana_local'] = (df['puntos_local'] > df['puntos_visitante']).astype(int)

print(f"✅ {len(df.columns)} características creadas")

# =====================================================================
# 4. SELECCIONAR CARACTERÍSTICAS Y LIMPIAR DATOS
# =====================================================================
print("\n📊 Preparando datos para modelo...")

caracteristicas = [
    'prom_puntos_local_anotados', 'prom_puntos_local_recibidos',
    'prom_puntos_vis_anotados', 'prom_puntos_vis_recibidos',
    'diferencia_puntos_local', 'diferencia_puntos_vis',
    'varianza_anotados_local', 'varianza_recibidos_local',
    'varianza_anotados_vis', 'varianza_recibidos_vis',
    'games_played_local', 'games_played_vis',
    'ratio_ofensivo', 'ratio_defensivo',
    'ventaja_ofensiva', 'ventaja_defensiva', 'experiencia_relativa'
]

# Reemplazar NaN e infinitos con 0
X = df[caracteristicas].fillna(0).replace([np.inf, -np.inf], 0)
y = df['gana_local']

print(f"✅ {len(caracteristicas)} características seleccionadas")
print(f"✅ Datos limpios y normalizados")

# =====================================================================
# 5. ANÁLISIS DE CARACTERÍSTICAS
# =====================================================================
print("\n📊 Análisis de correlación...")

# Correlación con el target
correlaciones = pd.DataFrame(X).corrwith(y)
correlaciones_abs = correlaciones.abs().sort_values(ascending=False)

print("\nTop 10 características por correlación:")
for i, (feat, corr) in enumerate(correlaciones_abs.head(10).items(), 1):
    print(f"   {i}. {feat}: {corr:.4f}")

# =====================================================================
# 6. DIVISIÓN INTELIGENTE DE DATOS
# =====================================================================
print("\n📊 División de datos para entrenamiento...")

# División temporal para evitar data leakage
punto_corte = int(0.8 * len(df))

X_train = X.iloc[:punto_corte]
y_train = y.iloc[:punto_corte]

X_test = X.iloc[punto_corte:]
y_test = y.iloc[punto_corte:]

print(f"   - Entrenamiento: {len(X_train)} partidos (80%)")
print(f"   - Evaluación: {len(X_test)} partidos (20%)")

# =====================================================================
# 7. ESCALADO (requerido para algunos modelos)
# =====================================================================
escalador = StandardScaler()
X_train_escalado = escalador.fit_transform(X_train)
X_test_escalado = escalador.transform(X_test)

# =====================================================================
# 8. ENTRENAMIENTO DE MÚLTIPLES MODELOS
# =====================================================================
print("\n🤖 Entrenando múltiples modelos...")

modelos = {}

# MODELO 1: Random Forest (potente, menos overfitting)
print("   1. Entrenando Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)
rf_model.fit(X_train, y_train)
modelos['Random Forest'] = rf_model

# MODELO 2: Gradient Boosting (muy poderoso)
print("   2. Entrenando Gradient Boosting...")
gb_model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    subsample=0.8
)
gb_model.fit(X_train, y_train)
modelos['Gradient Boosting'] = gb_model

# =====================================================================
# 9. EVALUACIÓN DE MODELOS
# =====================================================================
print("\n📈 ==================================================")
print("   EVALUACIÓN DE MODELOS")
print("==================================================\n")

mejor_accuracy = 0
mejor_modelo_nombre = ""
mejor_modelo = None

resultados = {}

for nombre, modelo in modelos.items():
    predicciones = modelo.predict(X_test)
    accuracy = accuracy_score(y_test, predicciones)
    
    # ROC-AUC para evaluación adicional
    predicciones_proba = modelo.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, predicciones_proba)
    
    resultados[nombre] = {
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'predicciones': predicciones,
        'probabilidades': predicciones_proba
    }
    
    print(f"🎯 {nombre}")
    print(f"   Accuracy: {accuracy:.2%}")
    print(f"   ROC-AUC: {roc_auc:.4f}")
    print(f"\n   Reporte de Clasificación:")
    print(classification_report(y_test, predicciones, target_names=['Gana Visitante', 'Gana Local']))
    print()
    
    if accuracy > mejor_accuracy:
        mejor_accuracy = accuracy
        mejor_modelo_nombre = nombre
        mejor_modelo = modelo

# =====================================================================
# 10. INFORMACIÓN DEL MEJOR MODELO
# =====================================================================
print(f"\n{'='*70}")
print(f"✅ MEJOR MODELO: {mejor_modelo_nombre}")
print(f"{'='*70}")
print(f"   Accuracy final: {mejor_accuracy:.2%}")
print(f"   Mejora respecto a baseline (61.86%): +{(mejor_accuracy - 0.6186)*100:.2f}%")
print(f"   ROC-AUC: {resultados[mejor_modelo_nombre]['roc_auc']:.4f}")

# =====================================================================
# 11. IMPORTANCIA DE CARACTERÍSTICAS (Feature Importance)
# =====================================================================
print(f"\n🔍 Características más importantes en {mejor_modelo_nombre}:")

if hasattr(mejor_modelo, 'feature_importances_'):
    importancias = mejor_modelo.feature_importances_
    indices_ordenados = np.argsort(importancias)[::-1]
    
    print("\n   Top 10 características:")
    for i, idx in enumerate(indices_ordenados[:10], 1):
        print(f"   {i}. {caracteristicas[idx]}: {importancias[idx]:.4f}")

# =====================================================================
# 12. GUARDAR MEJOR MODELO
# =====================================================================
print(f"\n💾 Guardando el mejor modelo ({mejor_modelo_nombre})...")
joblib.dump(mejor_modelo, 'mejor_modelo_predicciones.pkl')
joblib.dump(escalador, 'escalador_predicciones.pkl')
joblib.dump(caracteristicas, 'caracteristicas_modelo.pkl')
print("   ✅ Archivos guardados:")
print("      - mejor_modelo_predicciones.pkl")
print("      - escalador_predicciones.pkl")
print("      - caracteristicas_modelo.pkl")

# =====================================================================
# 13. RESUMEN FINAL
# =====================================================================
print(f"\n{'='*70}")
print("✅ RESUMEN DEL REENTRENAMIENTO OPTIMIZADO")
print(f"{'='*70}")
print(f"📊 Datos:")
print(f"   • Total de partidos: {len(df)}")
print(f"   • Características creadas: {len(caracteristicas)}")
print(f"\n🤖 Modelos entrenados:")
for nombre, resultado in resultados.items():
    print(f"   • {nombre}: {resultado['accuracy']:.2%}")
print(f"\n🏆 Mejor modelo: {mejor_modelo_nombre}")
print(f"   • Accuracy: {mejor_accuracy:.2%}")
print(f"   • ROC-AUC: {resultados[mejor_modelo_nombre]['roc_auc']:.4f}")
print(f"   • Mejora: +{(mejor_accuracy - 0.6186)*100:.2f}% vs baseline")
print(f"\n🚀 El modelo está listo para predicciones en tiempo real!")
print(f"{'='*70}\n")

# =====================================================================
# 14. INSTRUCCIONES PARA USAR EL MODELO GUARDADO
# =====================================================================
print("📝 Para usar el modelo guardado en futuras predicciones:")
print("""
    import joblib
    import pandas as pd
    
    modelo = joblib.load('mejor_modelo_predicciones.pkl')
    escalador = joblib.load('escalador_predicciones.pkl')
    caracteristicas = joblib.load('caracteristicas_modelo.pkl')
    
    # Preparar características igual que en entrenamiento
    X_nuevo = df_nuevo[caracteristicas].fillna(0).replace([np.inf, -np.inf], 0)
    
    prediccion = modelo.predict(X_nuevo)  # retorna 0 o 1
    probabilidades = modelo.predict_proba(X_nuevo)  # confianza
""")
