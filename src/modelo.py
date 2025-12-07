"""
RPSAI - Modelo de IA para Piedra, Papel o Tijera
=================================================
"""
import os
import pickle
import warnings
from pathlib import Path

import pandas as pd
import numpy as np

# --- Importaciones para Machine Learning ---
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Importamos los modelos y clases que vamos a optimizar
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# Descomentamos esta linea para ignorar el warning de sklearn
warnings.filterwarnings("ignore", message="X does not have valid feature names")
# Eliminamos la advertencia de zero_division en reportes
warnings.filterwarnings("ignore", message="Precision is ill-defined and being set to 0.0 in labels with no predicted samples.")


# =============================================================================
# CONFIGURACION Y CONSTANTES
# =============================================================================

RUTA_PROYECTO = Path(__file__).parent.parent
RUTA_DATOS = RUTA_PROYECTO / "data" / "partidas.csv"
RUTA_MODELO = RUTA_PROYECTO / "models" / "modelo_entrenado.pkl"

# Mapeo de jugadas a numeros (para el modelo)
JUGADA_A_NUM = {"piedra": 0, "papel": 1, "tijera": 2}
NUM_A_JUGADA = {0: "piedra", 1: "papel", 2: "tijera"}

# Que jugada gana a cual
PIERDE_CONTRA = {"piedra": "papel", "papel": "tijera", "tijera": "piedra"}
GANA_A = {"piedra": "tijera", "papel": "piedra", "tijera": "papel"}
TARGET_NAMES = list(JUGADA_A_NUM.keys()) # ['piedra', 'papel', 'tijera']


# =============================================================================
# PARTE 1: EXTRACCION Y PREPARACION DE DATOS
# =============================================================================

def cargar_datos(ruta_csv: Path = RUTA_DATOS) -> pd.DataFrame:
    """Carga los datos del CSV de partidas."""
    try:
        if not ruta_csv.exists():
            raise FileNotFoundError(f"No se encontró el archivo: {ruta_csv}")

        df = pd.read_csv(ruta_csv)
        columnas_requeridas = ['numero_ronda', 'jugada_j1', 'jugada_j2']
        if not all(col in df.columns for col in columnas_requeridas):
            raise ValueError("Faltan columnas requeridas.")

        print(f"✅ Datos cargados exitosamente: {len(df)} rondas")
        return df

    except Exception as e:
        print(f"❌ Error al cargar datos: {e}")
        return pd.DataFrame(columns=['numero_ronda', 'jugada_j1', 'jugada_j2'])


def preparar_datos(df: pd.DataFrame) -> pd.DataFrame:
    """Prepara los datos para el modelo y crea el target."""
    df = df.copy()

    # Convertir jugadas de texto a números
    df['jugada_j1_num'] = df['jugada_j1'].map(JUGADA_A_NUM)
    df['jugada_j2_num'] = df['jugada_j2'].map(JUGADA_A_NUM)

    # Target: próxima jugada del oponente (J2)
    df['proxima_jugada_j2'] = df['jugada_j2_num'].shift(-1)

    # Eliminar filas con NaN y convertir a enteros
    df = df.dropna()
    df['proxima_jugada_j2'] = df['proxima_jugada_j2'].astype(int)

    print(f"✅ Datos preparados: {len(df)} rondas válidas")
    return df


# =============================================================================
# PARTE 2: FEATURE ENGINEERING
# =============================================================================

def calcular_resultado(j1_num, j2_num):
    """Calcula el resultado de una ronda (1: J1 gana, 0: Empate, -1: J2 gana)."""
    if j1_num == j2_num:
        return 0
    elif (j1_num == 0 and j2_num == 2) or (j1_num == 1 and j2_num == 0) or (j1_num == 2 and j2_num == 1):
        return 1
    else:
        return -1


def crear_features(df: pd.DataFrame) -> pd.DataFrame:
    """Crea las features (caracteristicas) para el modelo."""
    df = df.copy()

    # Feature 1: Frecuencias de jugadas recientes (Ventana 3)
    df['frec_p_j2_3'] = (df['jugada_j2_num'] == 0).rolling(window=3, min_periods=1).mean().fillna(0.33)
    df['frec_a_j2_3'] = (df['jugada_j2_num'] == 1).rolling(window=3, min_periods=1).mean().fillna(0.33)
    df['frec_t_j2_3'] = (df['jugada_j2_num'] == 2).rolling(window=3, min_periods=1).mean().fillna(0.33)

    # Feature 2: Lag features (jugadas anteriores)
    df['j2_lag1'] = df['jugada_j2_num'].shift(1).fillna(1)
    df['j1_lag1'] = df['jugada_j1_num'].shift(1).fillna(1)

    # Feature 3: Resultado anterior
    df['resultado_anterior'] = df.apply(
        lambda row: calcular_resultado(row['jugada_j1_num'], row['jugada_j2_num']), axis=1
    ).shift(1).fillna(0)

    # Feature 4: Racha de J2 (pos: ganando, neg: perdiendo)
    df['racha_j2'] = 0
    for i in range(1, len(df)):
        if df['resultado_anterior'].iloc[i] == -1: # J2 ganó anterior
            df.iloc[i, df.columns.get_loc('racha_j2')] = max(0, df['racha_j2'].iloc[i-1]) + 1
        elif df['resultado_anterior'].iloc[i] == 1: # J2 perdió anterior
            df.iloc[i, df.columns.get_loc('racha_j2')] = min(0, df['racha_j2'].iloc[i-1]) - 1
        else:
            df.iloc[i, df.columns.get_loc('racha_j2')] = 0

    # Feature 5: Cambio en la jugada
    df['j2_cambio'] = (df['jugada_j2_num'] != df['jugada_j2_num'].shift(1)).astype(int).fillna(0)

    # Feature 6: Conteo de empates recientes (Ventana 5)
    df['empates_recientes'] = (df['resultado_anterior'] == 0).rolling(window=5, min_periods=1).sum().fillna(0)

    # Feature 7: Posición en el juego (normalizada)
    df['ronda_normalizada'] = df.index / len(df)

    print(f"✅ Features creadas: {len(df.columns)} columnas totales")
    return df


def seleccionar_features(df: pd.DataFrame) -> tuple:
    """Selecciona las features para entrenar y el target."""
    feature_cols = [
        'frec_p_j2_3', 'frec_a_j2_3', 'frec_t_j2_3',
        'j2_lag1', 'j1_lag1',
        'resultado_anterior', 'racha_j2', 'j2_cambio',
        'ronda_normalizada', 'empates_recientes'
        # Eliminamos features muy redundantes o complejas de calcular en la clase JugadorIA (ej. expanding mean)
    ]

    feature_cols = [col for col in feature_cols if col in df.columns]

    X = df[feature_cols]
    y = df['proxima_jugada_j2']

    print(f"✅ Features seleccionadas: {len(feature_cols)}")
    print(f"   X shape: {X.shape}, y shape: {y.shape}")

    return X, y


# =============================================================================
# PARTE 3: ENTRENAMIENTO Y OPTIMIZACIÓN
# =============================================================================

def entrenar_modelo(X, y, test_size: float = 0.2):
    """
    Entrena, evalua, optimiza y selecciona el mejor modelo.
    """
    # Dividir datos, manteniendo la secuencia (shuffle=False)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, shuffle=False
    )
    print(f"   Train size: {len(X_train)}, Test size: {len(X_test)}")

    resultados = []

    # --- 1. OPTIMIZACIÓN: KNN---
    print("\n" + "="*50)
    print("Entrenando: KNN")
    print("="*50)

    param_grid_knn = {
        'n_neighbors': [3, 5, 7, 9, 11, 15],  # Valores de k
        'weights': ['uniform', 'distance']
    }

    grid_search_knn = GridSearchCV(KNeighborsClassifier(), param_grid_knn, cv=5, scoring='accuracy')
    grid_search_knn.fit(X_train, y_train)

    best_knn = grid_search_knn.best_estimator_
    y_pred_knn = best_knn.predict(X_test)
    knn_accuracy = accuracy_score(y_test, y_pred_knn)

    print(f"Mejor K: {grid_search_knn.best_params_['n_neighbors']}, Peso: {grid_search_knn.best_params_['weights']}")
    print(f"Accuracy: {knn_accuracy:.4f}")
    print("Classification Report:\n", classification_report(y_test, y_pred_knn, target_names=TARGET_NAMES))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_knn))
    resultados.append(('KNN', knn_accuracy, best_knn))


    # --- 2. OPTIMIZACIÓN BÁSICA: DecisionTree ---
    print("\n" + "="*50)
    print("Entrenando: DecisionTree")
    print("="*50)
    # Reducimos la profundidad para prevenir el colapso (overfitting)
    dt = DecisionTreeClassifier(max_depth=5, min_samples_split=5, random_state=42)
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)
    dt_accuracy = accuracy_score(y_test, y_pred_dt)

    print(f"Accuracy: {dt_accuracy:.4f}")
    print("Classification Report:\n", classification_report(y_test, y_pred_dt, target_names=TARGET_NAMES))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_dt))
    resultados.append(('DecisionTree', dt_accuracy, dt))


    # --- 3. OPTIMIZACIÓN BÁSICA: RandomForest ---
    print("\n" + "="*50)
    print("Entrenando: RandomForest")
    print("="*50)
    # Usamos más estimadores y limitamos la profundidad
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_split=5, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    rf_accuracy = accuracy_score(y_test, y_pred_rf)

    print(f"Accuracy: {rf_accuracy:.4f}")
    print("Classification Report:\n", classification_report(y_test, y_pred_rf, target_names=TARGET_NAMES))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
    resultados.append(('RandomForest', rf_accuracy, rf))


    # --- SELECCION DEL MEJOR MODELO ---
    mejor_resultado = max(resultados, key=lambda x: x[1])
    mejor_nombre, mejor_accuracy, mejor_modelo = mejor_resultado

    print(f"\n" + "="*50)
    print(f"Mejor modelo: {mejor_nombre} con accuracy: {mejor_accuracy:.4f}")
    print(f"="*50)

    # Le añadimos los nombres de las features al modelo antes de guardarlo (para JugadorIA)
    mejor_modelo.feature_names_in_ = X.columns.tolist()

    return mejor_modelo


def guardar_modelo(modelo, ruta: Path = RUTA_MODELO):
    """Guarda el modelo entrenado en un archivo."""
    os.makedirs(ruta.parent, exist_ok=True)
    with open(ruta, "wb") as f:
        pickle.dump(modelo, f)
    print(f"✅ Modelo guardado en: {ruta}")


def cargar_modelo(ruta: Path = RUTA_MODELO):
    """Carga un modelo previamente entrenado."""
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontro el modelo en: {ruta}")
    with open(ruta, "rb") as f:
        return pickle.load(f)


# =============================================================================
# PARTE 4: PREDICCION Y JUEGO (Clase JugadorIA)
# =============================================================================

class JugadorIA:
    """Clase que encapsula el modelo para jugar."""

    def __init__(self, ruta_modelo: Path = RUTA_MODELO):
        """Inicializa el jugador IA."""
        self.modelo = None
        self.historial = []  # Lista de (jugada_j1, jugada_j2)
        self.feature_order = [] # Se cargará desde el modelo

        try:
            self.modelo = cargar_modelo(ruta_modelo)
            print("✅ Modelo cargado exitosamente.")

            # Aseguramos el orden de las features
            if hasattr(self.modelo, 'feature_names_in_'):
                self.feature_order = list(self.modelo.feature_names_in_)
                print(f"Features del modelo: {len(self.feature_order)}")

        except FileNotFoundError:
            print("⚠️ Modelo no encontrado. Usará estrategia aleatoria.")

    def registrar_ronda(self, jugada_j1: str, jugada_j2: str):
        """Registra una ronda jugada para actualizar el historial."""
        self.historial.append((jugada_j1, jugada_j2))

    def obtener_features_actuales(self) -> pd.DataFrame:
        """Genera las features basadas en el historial actual."""
        # Se necesita al menos 1 ronda para calcular el resultado anterior
        if not self.historial:
             # Retorna un DataFrame con valores por defecto (0 o 0.33)
            default_values = {col: 0.33 if 'frec' in col else 0 for col in self.feature_order}
            return pd.DataFrame([default_values])

        # Convertir historial a DataFrame temporal para calcular features de manera eficiente
        temp_df = pd.DataFrame(self.historial, columns=['jugada_j1', 'jugada_j2'])
        temp_df['jugada_j1_num'] = temp_df['jugada_j1'].map(JUGADA_A_NUM)
        temp_df['jugada_j2_num'] = temp_df['jugada_j2'].map(JUGADA_A_NUM)

        # Clonamos la lógica de crear_features para el historial

        # 1. Frecuencias de jugadas recientes (Ventana 3) - Usamos la última fila
        frec_p_j2_3 = (temp_df['jugada_j2_num'] == 0).rolling(window=3, min_periods=1).mean().iloc[-1]
        frec_a_j2_3 = (temp_df['jugada_j2_num'] == 1).rolling(window=3, min_periods=1).mean().iloc[-1]
        frec_t_j2_3 = (temp_df['jugada_j2_num'] == 2).rolling(window=3, min_periods=1).mean().iloc[-1]

        # 2. Lag features
        j2_lag1 = temp_df['jugada_j2_num'].iloc[-1]
        j1_lag1 = temp_df['jugada_j1_num'].iloc[-1]

        # 3. Resultado anterior
        if len(temp_df) >= 2:
            j1_ant = temp_df['jugada_j1_num'].iloc[-2]
            j2_ant = temp_df['jugada_j2_num'].iloc[-2]
            resultado_anterior = calcular_resultado(j1_ant, j2_ant)
        else:
            resultado_anterior = 0

        # 4. Racha (Requiere recalcular la racha completa en el historial)
        resultados_hist = temp_df.apply(lambda row: calcular_resultado(row['jugada_j1_num'], row['jugada_j2_num']), axis=1).shift(1).fillna(0)
        racha_j2 = 0
        for i in range(1, len(resultados_hist)):
            racha_anterior = racha_j2
            if resultados_hist.iloc[i] == -1: racha_j2 = max(0, racha_anterior) + 1
            elif resultados_hist.iloc[i] == 1: racha_j2 = min(0, racha_anterior) - 1
            else: racha_j2 = 0

        # 5. Cambio
        j2_cambio = 0
        if len(temp_df) >= 2:
            j2_cambio = 1 if temp_df['jugada_j2_num'].iloc[-1] != temp_df['jugada_j2_num'].iloc[-2] else 0

        # 6. Empates recientes
        empates_recientes = (resultados_hist.iloc[-5:] == 0).sum()

        # 7. Ronda normalizada
        ronda_normalizada = len(temp_df) / 300 # Asumiendo 300 como máximo para normalizar

        # Creamos el diccionario de features actual
        features = {
            'frec_p_j2_3': frec_p_j2_3, 'frec_a_j2_3': frec_a_j2_3, 'frec_t_j2_3': frec_t_j2_3,
            'j2_lag1': j2_lag1, 'j1_lag1': j1_lag1,
            'resultado_anterior': resultado_anterior, 'racha_j2': racha_j2, 'j2_cambio': j2_cambio,
            'ronda_normalizada': ronda_normalizada, 'empates_recientes': empates_recientes
        }

        df_features = pd.DataFrame([features])

        # Asegurar el orden de las features para el modelo
        return df_features[self.feature_order]


    def predecir_jugada_oponente(self) -> str:
        """Predice la proxima jugada del oponente."""
        if self.modelo is None:
            return np.random.choice(TARGET_NAMES)

        try:
            features_df = self.obtener_features_actuales()
            # Si el modelo espera 1D y obtiene 2D (caso de GridSearch), debe funcionar con .predict()
            prediccion_num = self.modelo.predict(features_df)[0]
            return NUM_A_JUGADA[prediccion_num]

        except Exception as e:
            # En caso de error (ej. feature mismatch), jugamos aleatorio para no colapsar
            print(f"⚠️ Error en predicción: {e}. Jugando aleatorio.")
            return np.random.choice(TARGET_NAMES)

    def decidir_jugada(self) -> str:
        """Decide que jugada hacer para ganar al oponente."""
        prediccion_oponente = self.predecir_jugada_oponente()
        return PIERDE_CONTRA[prediccion_oponente]


# =============================================================================
# FUNCION PRINCIPAL
# =============================================================================

def main():
    """Funcion principal para entrenar el modelo."""
    print("="*50)
    print("   RPSAI - Entrenamiento del Modelo")
    print("="*50)

    print("\n[1/6] Cargando datos...")
    df = cargar_datos()

    if df.empty or len(df) < 10:
        print("❌ No hay suficientes datos para entrenar.")
        return

    print("\n[2/6] Preparando datos...")
    df_preparado = preparar_datos(df)

    print("\n[3/6] Creando features...")
    df_features = crear_features(df_preparado)

    print("\n[4/6] Seleccionando features...")
    X, y = seleccionar_features(df_features)

    print("\n[5/6] Entrenando modelo (Optimización incluida)...")
    mejor_modelo = entrenar_modelo(X, y)

    print("\n[6/6] Guardando modelo...")
    guardar_modelo(mejor_modelo)

    print("\n✅ ¡Entrenamiento completado exitosamente!")


if __name__ == "__main__":
    main()