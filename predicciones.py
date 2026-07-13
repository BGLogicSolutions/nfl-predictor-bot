import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif
import xgboost as xgb
from catboost import CatBoostClassifier
import lightgbm as lgb
import joblib
import warnings
warnings.filterwarnings('ignore')

# =====================================================================
# 1. CARGAR DATOS DE TODAS LAS TEMPORADAS (2024 y 2025-2026)
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
print(f"   - Temporada 2024: {len(df_2024)} partidos")
print(f"   - Temporada 2025-2026: {len(df_2025_2026)} partidos")

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
# 6. ESCALADO
# =====================================================================
print("\n📊 Escalando características...")
escalador = StandardScaler()
X_escalado = escalador.fit_transform(X)
X = pd.DataFrame(X_escalado, columns=caracteristicas)

print(f"✅ Datos escalados correctamente")

# =====================================================================
# 7. ENTRENAMIENTO DE MÚLTIPLES MODELOS CON TODOS LOS DATOS
# =====================================================================
print("\n🤖 Entrenando múltiples modelos con TODOS los datos...")
print(f"   📊 Total de muestras: {len(X)}")
print(f"   📊 Balance: {y.mean():.2%} gana local\n")

modelos = {}

# MODELO 1: XGBoost
print("   1. Entrenando XGBoost...")
xgb_model = xgb.XGBClassifier(
    n_estimators=400,
    max_depth=7,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss',
    verbosity=0
)
xgb_model.fit(X, y)
modelos['XGBoost'] = xgb_model

# MODELO 2: CatBoost
print("   2. Entrenando CatBoost...")
cat_model = CatBoostClassifier(
    iterations=400,
    learning_rate=0.03,
    depth=7,
    loss_function='Logloss',
    verbose=0,
    random_state=42
)
cat_model.fit(X, y)
modelos['CatBoost'] = cat_model

# MODELO 3: LightGBM
print("   3. Entrenando LightGBM...")
lgb_model = lgb.LGBMClassifier(
    n_estimators=400,
    max_depth=7,
    learning_rate=0.03,
    num_leaves=63,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1
)
lgb_model.fit(X, y)
modelos['LightGBM'] = lgb_model

# MODELO 4: Gradient Boosting
print("   4. Entrenando Gradient Boosting...")
gb_model = GradientBoostingClassifier(
    n_estimators=400,
    learning_rate=0.03,
    max_depth=6,
    min_samples_split=5,
    min_samples_leaf=2,
    subsample=0.8,
    random_state=42
)
gb_model.fit(X, y)
modelos['Gradient Boosting'] = gb_model

# MODELO 5: Random Forest
print("   5. Entrenando Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)
rf_model.fit(X, y)
modelos['Random Forest'] = rf_model

# =====================================================================
# 8. VALIDACIÓN CRUZADA ESTRATIFICADA (5-FOLD)
# =====================================================================
print("\n📈 ==================================================")
print("   VALIDACIÓN CRUZADA (5-FOLD STRATIFIED)")
print("==================================================\n")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

resultados = {}

for nombre, modelo in modelos.items():
    print(f"🎯 {nombre}")
    
    # Cross-validation scores
    cv_scores = cross_val_score(modelo, X, y, cv=skf, scoring='accuracy')
    
    resultados[nombre] = {
        'cv_scores': cv_scores,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'modelo': modelo
    }
    
    print(f"   Scores (5-fold): {[f'{s:.2%}' for s in cv_scores]}")
    print(f"   Media: {cv_scores.mean():.2%}")
    print(f"   Std Dev: {cv_scores.std():.4f}")
    print(f"   Min: {cv_scores.min():.2%} | Max: {cv_scores.max():.2%}")
    print()

# =====================================================================
# 9. ENSEMBLE VOTING CON FIT
# =====================================================================
print("📊 Creando Ensemble Voting (XGBoost + CatBoost + LightGBM)...\n")

voting_clf = VotingClassifier(
    estimators=[
        ('xgb', xgb_model),
        ('cat', cat_model),
        ('lgb', lgb_model)
    ],
    voting='soft'
)

# Entrenar ensemble
voting_clf.fit(X, y)

# Validación cruzada del ensemble
print(f"🎯 ENSEMBLE VOTING")
ensemble_cv_scores = cross_val_score(voting_clf, X, y, cv=skf, scoring='accuracy')

resultados['Ensemble Voting'] = {
    'cv_scores': ensemble_cv_scores,
    'cv_mean': ensemble_cv_scores.mean(),
    'cv_std': ensemble_cv_scores.std(),
    'modelo': voting_clf
}

print(f"   Scores (5-fold): {[f'{s:.2%}' for s in ensemble_cv_scores]}")
print(f"   Media: {ensemble_cv_scores.mean():.2%}")
print(f"   Std Dev: {ensemble_cv_scores.std():.4f}")
print(f"   Min: {ensemble_cv_scores.min():.2%} | Max: {ensemble_cv_scores.max():.2%}")
print()

# =====================================================================
# 10. SELECCIONAR MEJOR MODELO
# =====================================================================
print("="*70)
print("📊 RESUMEN DE RESULTADOS")
print("="*70 + "\n")

mejor_accuracy = 0
mejor_modelo_nombre = ""
mejor_modelo = None

print("🏆 Ranking de modelos por Accuracy (5-fold CV):\n")
ranked = sorted(resultados.items(), key=lambda x: x[1]['cv_mean'], reverse=True)

for i, (nombre, data) in enumerate(ranked, 1):
    print(f"   {i}. {nombre}")
    print(f"      Accuracy: {data['cv_mean']:.2%} (±{data['cv_std']:.2%})")
    
    if data['cv_mean'] > mejor_accuracy:
        mejor_accuracy = data['cv_mean']
        mejor_modelo_nombre = nombre
        mejor_modelo = data['modelo']

print(f"\n{'='*70}")
print(f"✅ MEJOR MODELO: {mejor_modelo_nombre}")
print(f"   Accuracy CV: {mejor_accuracy:.2%}")
print(f"   Mejora vs baseline (61.86%): +{(mejor_accuracy - 0.6186)*100:.2f}%")
print(f"{'='*70}\n")

# =====================================================================
# 11. IMPORTANCIA DE CARACTERÍSTICAS
# =====================================================================
print("🔍 Características más importantes en el mejor modelo:\n")

if hasattr(mejor_modelo, 'feature_importances_'):
    importancias = mejor_modelo.feature_importances_
elif hasattr(mejor_modelo, 'get_feature_importance'):
    importancias = mejor_modelo.get_feature_importance()
else:
    # Para Voting, usar el primer estimador
    try:
        importancias = mejor_modelo.estimators_[0].feature_importances_
    except:
        importancias = None

if importancias is not None:
    indices_ordenados = np.argsort(importancias)[::-1]
    
    print("   Top 10 características:")
    for i, idx in enumerate(indices_ordenados[:10], 1):
        print(f"   {i}. {caracteristicas[idx]}: {importancias[idx]:.4f}")
else:
    print("   (Feature importance no disponible para este modelo)")

# =====================================================================
# 12. GUARDAR MODELO
# =====================================================================
print(f"\n💾 Guardando mejor modelo ({mejor_modelo_nombre})...")
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
print("✅ RESUMEN FINAL DEL REENTRENAMIENTO")
print(f"{'='*70}")
print(f"\n📊 Datos:")
print(f"   • Total de partidos: {len(df)}")
print(f"   • Temporada 2024: {len(df_2024)}")
print(f"   • Temporada 2025-2026: {len(df_2025_2026)}")
print(f"   • Características iniciales: {len(todas_caracteristicas)}")
print(f"   • Características después de SelectKBest: {len(caracteristicas)}")
print(f"\n🤖 Modelos entrenados:")
for nombre, data in ranked:
    print(f"   • {nombre}: {data['cv_mean']:.2%}")
print(f"\n🏆 MEJOR MODELO: {mejor_modelo_nombre}")
print(f"   • Accuracy (5-fold CV): {mejor_accuracy:.2%}")
print(f"   • Desviación estándar: ±{resultados[mejor_modelo_nombre]['cv_std']:.2%}")
print(f"   • Mejora vs baseline: +{(mejor_accuracy - 0.6186)*100:.2f}%")
print(f"\n🚀 ¡El modelo está listo para predicciones en producción!")
print(f"{'='*70}\n")
