import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif
import xgboost as xgb
from catboost import CatBoostClassifier
import lightgbm as lgb
import joblib
import warnings
warnings.filterwarnings('ignore')

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
historial_ganaos = {equipo: [] for equipo in equipos}

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
df['win_rate_local'] = 0.5
df['win_rate_vis'] = 0.5

print("   Calculando promedios móviles, varianzas, win rates y tendencias...")

for idx, row in df.iterrows():
    loc = row['equipo_local']
    vis = row['equipo_visitante']
    
    # PROMEDIO MÓVIL
    if historial_anotados[loc]:
        df.at[idx, 'prom_puntos_local_anotados'] = np.mean(historial_anotados[loc])
        df.at[idx, 'prom_puntos_local_recibidos'] = np.mean(historial_recibidos[loc])
        df.at[idx, 'varianza_anotados_local'] = np.var(historial_anotados[loc]) if len(historial_anotados[loc]) > 1 else 0
        df.at[idx, 'varianza_recibidos_local'] = np.var(historial_recibidos[loc]) if len(historial_recibidos[loc]) > 1 else 0
        df.at[idx, 'games_played_local'] = len(historial_anotados[loc])
        df.at[idx, 'win_rate_local'] = np.mean(historial_ganaos[loc]) if historial_ganaos[loc] else 0.5
        
    if historial_anotados[vis]:
        df.at[idx, 'prom_puntos_vis_anotados'] = np.mean(historial_anotados[vis])
        df.at[idx, 'prom_puntos_vis_recibidos'] = np.mean(historial_recibidos[vis])
        df.at[idx, 'varianza_anotados_vis'] = np.var(historial_anotados[vis]) if len(historial_anotados[vis]) > 1 else 0
        df.at[idx, 'varianza_recibidos_vis'] = np.var(historial_recibidos[vis]) if len(historial_recibidos[vis]) > 1 else 0
        df.at[idx, 'games_played_vis'] = len(historial_anotados[vis])
        df.at[idx, 'win_rate_vis'] = np.mean(historial_ganaos[vis]) if historial_ganaos[vis] else 0.5
    
    # DIFERENCIA (métrica clave)
    df.at[idx, 'diferencia_puntos_local'] = df.at[idx, 'prom_puntos_local_anotados'] - df.at[idx, 'prom_puntos_local_recibidos']
    df.at[idx, 'diferencia_puntos_vis'] = df.at[idx, 'prom_puntos_vis_anotados'] - df.at[idx, 'prom_puntos_vis_recibidos']
    
    # Guardar resultados
    historial_anotados[loc].append(row['puntos_local'])
    historial_recibidos[loc].append(row['puntos_visitante'])
    historial_ganaos[loc].append(1 if row['puntos_local'] > row['puntos_visitante'] else 0)
    
    historial_anotados[vis].append(row['puntos_visitante'])
    historial_recibidos[vis].append(row['puntos_local'])
    historial_ganaos[vis].append(1 if row['puntos_visitante'] > row['puntos_local'] else 0)

# CARACTERÍSTICAS DERIVADAS (interacciones y ratios)
df['ratio_ofensivo'] = df['prom_puntos_local_anotados'] / (df['prom_puntos_vis_recibidos'] + 1)
df['ratio_defensivo'] = df['prom_puntos_local_recibidos'] / (df['prom_puntos_vis_anotados'] + 1)
df['ventaja_ofensiva'] = df['prom_puntos_local_anotados'] - df['prom_puntos_vis_recibidos']
df['ventaja_defensiva'] = df['prom_puntos_vis_anotados'] - df['prom_puntos_local_recibidos']
df['experiencia_relativa'] = df['games_played_local'] - df['games_played_vis']
df['diferencial_win_rate'] = df['win_rate_local'] - df['win_rate_vis']

# Momentos (últimos 3 partidos)
df['momentum_local'] = 0.0
df['momentum_vis'] = 0.0

historial_anotados = {equipo: [] for equipo in equipos}
historial_recibidos = {equipo: [] for equipo in equipos}

for idx, row in df.iterrows():
    loc = row['equipo_local']
    vis = row['equipo_visitante']
    
    if historial_anotados[loc] and len(historial_anotados[loc]) >= 3:
        df.at[idx, 'momentum_local'] = np.mean(historial_anotados[loc][-3:]) - np.mean(historial_anotados[loc][:-3] if len(historial_anotados[loc]) > 3 else historial_anotados[loc])
    
    if historial_anotados[vis] and len(historial_anotados[vis]) >= 3:
        df.at[idx, 'momentum_vis'] = np.mean(historial_anotados[vis][-3:]) - np.mean(historial_anotados[vis][:-3] if len(historial_anotados[vis]) > 3 else historial_anotados[vis])
    
    historial_anotados[loc].append(row['puntos_local'])
    historial_anotados[vis].append(row['puntos_visitante'])

# Variable objetivo
df['gana_local'] = (df['puntos_local'] > df['puntos_visitante']).astype(int)

print(f"✅ {len(df.columns)} características creadas")

# =====================================================================
# 4. SELECCIONAR Y LIMPIAR CARACTERÍSTICAS
# =====================================================================
print("\n📊 Preparando datos para modelo...")

todas_caracteristicas = [
    'prom_puntos_local_anotados', 'prom_puntos_local_recibidos',
    'prom_puntos_vis_anotados', 'prom_puntos_vis_recibidos',
    'diferencia_puntos_local', 'diferencia_puntos_vis',
    'varianza_anotados_local', 'varianza_recibidos_local',
    'varianza_anotados_vis', 'varianza_recibidos_vis',
    'games_played_local', 'games_played_vis',
    'ratio_ofensivo', 'ratio_defensivo',
    'ventaja_ofensiva', 'ventaja_defensiva', 'experiencia_relativa',
    'win_rate_local', 'win_rate_vis', 'diferencial_win_rate',
    'momentum_local', 'momentum_vis'
]

X = df[todas_caracteristicas].fillna(0).replace([np.inf, -np.inf], 0)
y = df['gana_local']

print(f"✅ {len(todas_caracteristicas)} características seleccionadas")

# =====================================================================
# 5. FEATURE SELECTION - ELIMINAR CARACTERÍSTICAS RUIDOSAS
# =====================================================================
print("\n🔍 Seleccionando mejores características (SelectKBest)...")

selector = SelectKBest(f_classif, k=15)
X_selected = selector.fit_transform(X, y)

# Obtener nombres de características seleccionadas
mascara = selector.get_support()
caracteristicas = [todas_caracteristicas[i] for i in range(len(todas_caracteristicas)) if mascara[i]]

print(f"✅ Características seleccionadas ({len(caracteristicas)}):")
for i, feat in enumerate(caracteristicas, 1):
    print(f"   {i}. {feat}")

X = pd.DataFrame(X_selected, columns=caracteristicas)

# =====================================================================
# 6. DIVISIÓN INTELIGENTE DE DATOS
# =====================================================================
print("\n📊 División de datos para entrenamiento...")

punto_corte = int(0.8 * len(df))

X_train = X.iloc[:punto_corte].reset_index(drop=True)
y_train = y.iloc[:punto_corte].reset_index(drop=True)

X_test = X.iloc[punto_corte:].reset_index(drop=True)
y_test = y.iloc[punto_corte:].reset_index(drop=True)

print(f"   - Entrenamiento: {len(X_train)} partidos (80%)")
print(f"   - Evaluación: {len(X_test)} partidos (20%)")
print(f"   - Balance entrenamiento: {y_train.mean():.2%} gana local")
print(f"   - Balance evaluación: {y_test.mean():.2%} gana local")

# =====================================================================
# 7. ESCALADO
# =====================================================================
escalador = StandardScaler()
X_train_escalado = escalador.fit_transform(X_train)
X_test_escalado = escalador.transform(X_test)

# =====================================================================
# 8. ENTRENAMIENTO DE MÚLTIPLES MODELOS AVANZADOS
# =====================================================================
print("\n🤖 Entrenando múltiples modelos avanzados...\n")

modelos = {}

# MODELO 1: XGBoost (muy poderoso)
print("   1. Entrenando XGBoost...")
xgb_model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss',
    verbosity=0
)
xgb_model.fit(X_train, y_train)
modelos['XGBoost'] = xgb_model

# MODELO 2: CatBoost (excelente con datos categóricos)
print("   2. Entrenando CatBoost...")
cat_model = CatBoostClassifier(
    iterations=300,
    learning_rate=0.05,
    depth=6,
    loss_function='Logloss',
    verbose=0,
    random_state=42
)
cat_model.fit(X_train, y_train)
modelos['CatBoost'] = cat_model

# MODELO 3: LightGBM (rápido y preciso)
print("   3. Entrenando LightGBM...")
lgb_model = lgb.LGBMClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1
)
lgb_model.fit(X_train, y_train)
modelos['LightGBM'] = lgb_model

# MODELO 4: Gradient Boosting con parámetros optimizados
print("   4. Entrenando Gradient Boosting optimizado...")
gb_model = GradientBoostingClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    min_samples_split=5,
    min_samples_leaf=2,
    subsample=0.8,
    random_state=42
)
gb_model.fit(X_train, y_train)
modelos['Gradient Boosting'] = gb_model

# =====================================================================
# 9. EVALUACIÓN DE MODELOS INDIVIDUALES
# =====================================================================
print("\n📈 ==================================================")
print("   EVALUACIÓN DE MODELOS INDIVIDUALES")
print("==================================================\n")

mejor_accuracy = 0
mejor_modelo_nombre = ""
mejor_modelo = None

resultados = {}

for nombre, modelo in modelos.items():
    predicciones = modelo.predict(X_test)
    accuracy = accuracy_score(y_test, predicciones)
    
    predicciones_proba = modelo.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, predicciones_proba)
    
    resultados[nombre] = {
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'predicciones': predicciones,
        'probabilidades': predicciones_proba,
        'modelo': modelo
    }
    
    print(f"🎯 {nombre}")
    print(f"   Accuracy: {accuracy:.2%}")
    print(f"   ROC-AUC: {roc_auc:.4f}")
    print()
    
    if accuracy > mejor_accuracy:
        mejor_accuracy = accuracy
        mejor_modelo_nombre = nombre
        mejor_modelo = modelo

# =====================================================================
# 10. ENSEMBLE VOTING - COMBINAR MEJORES MODELOS
# =====================================================================
print("📊 Creando Ensemble Voting con mejores modelos...\n")

# Usar los 3 mejores modelos en voting
voting_clf = VotingClassifier(
    estimators=[
        ('xgb', modelos['XGBoost']),
        ('cat', modelos['CatBoost']),
        ('lgb', modelos['LightGBM'])
    ],
    voting='soft'
)

predicciones_ensemble = voting_clf.predict(X_test)
accuracy_ensemble = accuracy_score(y_test, predicciones_ensemble)

probabilidades_ensemble = voting_clf.predict_proba(X_test)[:, 1]
roc_auc_ensemble = roc_auc_score(y_test, probabilidades_ensemble)

print(f"🎯 ENSEMBLE VOTING (XGBoost + CatBoost + LightGBM)")
print(f"   Accuracy: {accuracy_ensemble:.2%}")
print(f"   ROC-AUC: {roc_auc_ensemble:.4f}")
print()

if accuracy_ensemble > mejor_accuracy:
    mejor_accuracy = accuracy_ensemble
    mejor_modelo_nombre = "Ensemble Voting"
    mejor_modelo = voting_clf
    resultados['Ensemble Voting'] = {
        'accuracy': accuracy_ensemble,
        'roc_auc': roc_auc_ensemble,
        'predicciones': predicciones_ensemble,
        'probabilidades': probabilidades_ensemble,
        'modelo': voting_clf
    }

# =====================================================================
# 11. VALIDACIÓN CRUZADA DEL MEJOR MODELO
# =====================================================================
print("📊 ==================================================")
print("   VALIDACIÓN CRUZADA DEL MEJOR MODELO")
print("==================================================\n")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(mejor_modelo, X_train, y_train, cv=skf, scoring='accuracy')

print(f"🎯 {mejor_modelo_nombre}")
print(f"   Scores CV (5-fold): {cv_scores}")
print(f"   Media: {cv_scores.mean():.2%}")
print(f"   Std Dev: {cv_scores.std():.4f}")
print(f"   Min: {cv_scores.min():.2%}")
print(f"   Max: {cv_scores.max():.2%}")

# =====================================================================
# 12. REPORTE FINAL
# =====================================================================
print(f"\n{'='*70}")
print("📊 REPORTE DE CLASIFICACIÓN - MEJOR MODELO")
print(f"{'='*70}\n")

print(classification_report(y_test, resultados[mejor_modelo_nombre]['predicciones'], 
                          target_names=['Gana Visitante', 'Gana Local']))

print("\n" + "="*70)
print("✅ RESUMEN FINAL DEL REENTRENAMIENTO")
print("="*70)
print(f"📊 Datos:")
print(f"   • Total de partidos: {len(df)}")
print(f"   • Características después de Feature Selection: {len(caracteristicas)}")
print(f"\n🤖 Modelos individuales entrenados:")
for nombre, resultado in resultados.items():
    if nombre != 'Ensemble Voting':
        print(f"   • {nombre}: {resultado['accuracy']:.2%}")

print(f"\n🏆 MEJOR MODELO: {mejor_modelo_nombre}")
print(f"   • Accuracy: {mejor_accuracy:.2%}")
print(f"   • ROC-AUC: {resultados[mejor_modelo_nombre]['roc_auc']:.4f}")
print(f"   • Mejora vs baseline (61.86%): +{(mejor_accuracy - 0.6186)*100:.2f}%")
print(f"\n💾 Guardando modelo...")

joblib.dump(mejor_modelo, 'mejor_modelo_predicciones.pkl')
joblib.dump(escalador, 'escalador_predicciones.pkl')
joblib.dump(caracteristicas, 'caracteristicas_modelo.pkl')

print("   ✅ Archivos guardados:")
print("      - mejor_modelo_predicciones.pkl")
print("      - escalador_predicciones.pkl")
print("      - caracteristicas_modelo.pkl")

print(f"\n🚀 ¡El modelo optimizado está listo para producción!")
print("="*70 + "\n")
