# 🏈 NFL Predictor Bot

Sistema de predicción de resultados de partidos NFL utilizando Machine Learning avanzado con ensemble de modelos.

## 📊 Descripción

Este proyecto entrena múltiples modelos de Machine Learning (XGBoost, CatBoost, LightGBM, Gradient Boosting, Random Forest) para predecir el resultado de partidos de la NFL. Los datos se combinan de las temporadas 2024 y 2025-2026, creando características avanzadas basadas en:

- Promedios móviles de puntos anotados y recibidos
- Tasas de victoria (win rates)
- Ventajas ofensivas y defensivas
- Ratios de rendimiento
- Momentum reciente
- Varianzas de consistencia

## 🎯 Rendimiento del Modelo

**Mejor Modelo: Ensemble Voting (XGBoost + CatBoost + LightGBM)**

- **Accuracy (5-fold CV)**: 75%+ 
- **Mejora vs Baseline**: +13-20%
- **ROC-AUC**: 0.80+
- **Datos de Entrenamiento**: 590 partidos (2024 + 2025-2026)
- **Características**: 15 seleccionadas de 22 iniciales

## 📁 Estructura del Proyecto

```
nfl-predictor-bot/
├── predicciones.py                    # Script principal de entrenamiento
├── mejor_modelo_predicciones.pkl      # Modelo entrenado (Ensemble)
├── escalador_predicciones.pkl         # Escalador StandardScaler
├── caracteristicas_modelo.pkl         # Lista de características
├── temporada_2024_crudo.csv           # Datos de 2024
├── temporada_2025_2026_crudo.csv      # Datos de 2025-2026
├── requirements.txt                   # Dependencias
└── README.md                          # Este archivo
```

## 🚀 Instalación

1. **Clonar el repositorio**:
```bash
git clone https://github.com/BGLogicSolutions/nfl-predictor-bot.git
cd nfl-predictor-bot
```

2. **Instalar dependencias**:
```bash
pip install -r requirements.txt
```

## 🔧 Uso

### Para Reentrenar el Modelo

```bash
python predicciones.py
```

Esto:
- Carga datos de 2024 y 2025-2026
- Crea características avanzadas
- Selecciona las mejores características (SelectKBest)
- Entrena 5 modelos diferentes
- Realiza validación cruzada 5-fold
- Crea Ensemble Voting
- Guarda el mejor modelo
- Genera reporte detallado

### Para Hacer Predicciones

```python
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Cargar modelo y recursos
modelo = joblib.load('mejor_modelo_predicciones.pkl')
escalador = joblib.load('escalador_predicciones.pkl')
caracteristicas = joblib.load('caracteristicas_modelo.pkl')

# Preparar datos (debe tener las mismas características)
X_nuevo = pd.DataFrame(...)[caracteristicas]  # 15 características
X_nuevo = X_nuevo.fillna(0).replace([np.inf, -np.inf], 0)
X_escalado = escalador.transform(X_nuevo)

# Hacer predicción
prediccion = modelo.predict(X_escalado)  # 0: Gana Visitante, 1: Gana Local
probabilidades = modelo.predict_proba(X_escalado)  # Confianza de predicción

print(f"Predicción: {'Gana Local' if prediccion[0] == 1 else 'Gana Visitante'}")
print(f"Confianza: {max(probabilidades[0])*100:.2f}%")
```

## 📊 Características Utilizadas

El modelo utiliza 15 características seleccionadas automáticamente:

1. **prom_puntos_local_anotados** - Promedio de puntos anotados por local
2. **prom_puntos_local_recibidos** - Promedio de puntos recibidos por local
3. **prom_puntos_vis_anotados** - Promedio de puntos anotados por visitante
4. **diferencia_puntos_local** - Diferencial ofensivo-defensivo local
5. **diferencia_puntos_vis** - Diferencial ofensivo-defensivo visitante
6. **varianza_anotados_local** - Consistencia anotadora local
7. **varianza_anotados_vis** - Consistencia anotadora visitante
8. **ratio_ofensivo** - Ratio ofensivo (local/defensa visitante)
9. **ratio_defensivo** - Ratio defensivo (defensa local/ataque visitante)
10. **ventaja_ofensiva** - Ventaja ofensiva relativa
11. **experiencia_relativa** - Diferencia en partidos jugados
12. **win_rate_local** - Tasa de victorias local
13. **win_rate_vis** - Tasa de victorias visitante
14. **diferencial_win_rate** - Diferencia en tasas de victoria
15. **momentum_vis** - Momentum reciente del visitante

## 🤖 Modelos Implementados

### Modelos Individuales
- **XGBoost**: Gradient boosting con optimización avanzada
- **CatBoost**: Optimizado para variables categóricas
- **LightGBM**: Boosting rápido y eficiente
- **Gradient Boosting**: Scikit-learn clásico mejorado
- **Random Forest**: Ensemble de árboles de decisión

### Modelo Final
- **Ensemble Voting**: Combinación soft de XGBoost + CatBoost + LightGBM
- Utiliza votación ponderada para mayor robustez
- Mejor generalización

## 📈 Metodología

### 1. Ingeniería de Características
- Cálculo de promedios móviles acumulados
- Varianzas para medir consistencia
- Ratios y ventajas relativas
- Momentum (últimos 3 partidos)
- Tasas de victoria

### 2. Selección de Características
- **SelectKBest** con scoring f_classif
- Reduce de 22 a 15 características más relevantes
- Elimina ruido y mejora generalización

### 3. Entrenamiento
- Escalado con StandardScaler
- Validación cruzada 5-fold estratificada
- Uso de TODOS los datos (590 partidos)
- Balanceo de clases en modelos

### 4. Evaluación
- Accuracy en validación cruzada
- ROC-AUC para calibración
- Matriz de confusión
- Importancia de características

## 📚 Dependencias

```
requests>=2.25.0
pandas>=1.3.0
scikit-learn>=0.24.0
matplotlib>=3.3.0
seaborn>=0.11.0
joblib>=1.0.0
numpy>=1.19.0
xgboost>=1.5.0
catboost>=1.0.0
lightgbm>=3.0.0
```

## 🎯 Mejoras Futuras

- [ ] Agregar datos de lesiones de jugadores
- [ ] Incluir estadísticas H2H (head-to-head)
- [ ] Considerar coaching changes
- [ ] Modelar factores ambientales (clima, altitud)
- [ ] Actualizar modelo mensualmente con nuevos datos
- [ ] API REST para predicciones en tiempo real
- [ ] Dashboard web de visualización
- [ ] Sistema de apuestas recomendadas

## 📝 Datos

### Estructura CSV
```
id_partido | semana | equipo_local | equipo_visitante | puntos_local | puntos_visitante
```

### Fuentes
- **2024**: Temporada completa NFL
- **2025-2026**: Temporada en progreso

## 🔍 Análisis de Resultados

Ejecuta `predicciones.py` para ver:
- Accuracy de cada modelo
- Validación cruzada 5-fold
- Ranking de modelos
- Importancia de características
- Reporte de clasificación detallado

## 💡 Ejemplos de Uso

### Predicción Simple
```python
import joblib

modelo = joblib.load('mejor_modelo_predicciones.pkl')
prediccion = modelo.predict([[20, 20, 20, 20, 0, 0, 0, 1, 1, 5, 0, 0.5, 0.5, 0, 0]])
print("Gana Local" if prediccion[0] == 1 else "Gana Visitante")
```

### Con Probabilidades
```python
prob = modelo.predict_proba([[...]])
print(f"Probabilidad: {prob[0][1]*100:.2f}%")
```

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:
1. Fork el repositorio
2. Crea una rama para tu feature
3. Haz commit de tus cambios
4. Push a la rama
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está bajo licencia MIT.

## 👤 Autor

**BGLogicSolutions**

## 📞 Soporte

Para reportar bugs o sugerencias, abre un issue en GitHub.

---

**Última actualización**: Julio 13, 2026
**Versión**: 2.0 - Ensemble Avanzado
